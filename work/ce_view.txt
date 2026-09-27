"""Pair cross-encoder: intfloat/multilingual-e5-small (MIT, 118M) fine-tuned as a binary classifier
on raw '<name> | <address>' of (S1, query) pairs. Its logit becomes a LightGBM feature (ce_logit).

Leakage control: train queries are split in two folds by hash. Cross-encoder "a" trains on fold A,
cross-encoder "b" on fold B (cross-fitting). Each train query gets the logit of the model that did
not see it; validation (held-out S1) and test pairs get the mean of both logits.
"""
import math
import sys
import time

import numpy as np
import polars as pl
import torch

from .config import SEED, WORK_DIR
from .io_utils import load_source

BASE = "intfloat/multilingual-e5-small"
CE_DIRS = {"a": WORK_DIR / "ce_e5s", "b": WORK_DIR / "ce_e5s_b"}
MAX_LEN = 128
N_TRAIN_PAIRS = 1_500_000


def fold_b(col: str = "q_row"):
    """Fold B (LightGBM training) vs fold A (cross-encoder training) of the train queries."""
    return (pl.col(col).hash(seed=SEED + 1) % 2) == 1


def _texts(split: str):
    def t(s):
        d = load_source(split, s)
        return (d["business_name"].fill_null("") + " | " + d["business_address"].fill_null("")).to_list()
    return t(1), t(2) + t(3)


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE)


def _encode(tok, a: list, b: list) -> list:
    return tok(a, b, truncation="longest_first", max_length=MAX_LEN)["input_ids"]


def _collate(ids: list, pad: int, device: str):
    n = max(len(x) for x in ids)
    arr = np.full((len(ids), n), pad, dtype=np.int64)
    for i, x in enumerate(ids):
        arr[i, :len(x)] = x
    t = torch.from_numpy(arr).to(device, non_blocking=True)
    return t, (t != pad).long()


def train_ce(pairs: pl.DataFrame, out_dir, split: str = "train", epochs: int = 1, bs: int = 128, lr: float = 5e-5):
    """pairs: q_row, s1_row, label."""
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    tok = _tokenizer()
    s1_txt, q_txt = _texts(split)
    pairs = pairs.sample(fraction=1.0, shuffle=True, seed=SEED)
    t = time.time()
    ids = _encode(tok, [s1_txt[i] for i in pairs["s1_row"].to_list()], [q_txt[i] for i in pairs["q_row"].to_list()])
    y = pairs["label"].to_numpy().astype(np.float32)
    print(f"  ce tokenized {len(ids):,} pairs {time.time() - t:.0f}s", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(ids) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    step, t, run = 0, time.time(), 0.0
    for ep in range(epochs):
        order = np.random.default_rng(SEED + ep).permutation(len(ids))
        for i in range(0, len(ids), bs):
            bi = order[i:i + bs]
            x, m = _collate([ids[j] for j in bi], tok.pad_token_id, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(input_ids=x, attention_mask=m).logits.squeeze(-1)
            loss = lossf(logit.float(), torch.from_numpy(y[bi]).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
            step += 1
            if step % 500 == 0:
                print(f"  ce step {step}/{steps} loss {run:.4f} {(step * bs) / (time.time() - t):.0f} pairs/s", flush=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return model


@torch.no_grad()
def score_ce(pairs: pl.DataFrame, split: str, fold: str = "a", bs: int = 512, chunk: int = 1_000_000) -> np.ndarray:
    """ce_logit for each (q_row, s1_row) row, in input order. Length-sorted batches for speed."""
    from transformers import AutoModelForSequenceClassification
    tok = _tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(CE_DIRS[fold]).cuda()
    model.eval().half()
    s1_txt, q_txt = _texts(split)
    out = np.empty(len(pairs), dtype=np.float32)
    s1r, qr = pairs["s1_row"].to_numpy(), pairs["q_row"].to_numpy()
    t = time.time()
    for c in range(0, len(pairs), chunk):
        ids = _encode(tok, [s1_txt[i] for i in s1r[c:c + chunk]], [q_txt[i] for i in qr[c:c + chunk]])
        order = np.argsort([len(x) for x in ids], kind="stable")
        res = np.empty(len(ids), dtype=np.float32)
        for i in range(0, len(ids), bs):
            bi = order[i:i + bs]
            x, m = _collate([ids[j] for j in bi], tok.pad_token_id, "cuda")
            res[bi] = model(input_ids=x, attention_mask=m).logits.squeeze(-1).float().cpu().numpy()
        out[c:c + chunk] = res
        print(f"  ce score [{split}]: {min(c + chunk, len(pairs)):,}/{len(pairs):,} {time.time() - t:.0f}s", flush=True)
    return out


def ce_scores(split: str, fold: str = "a", force: bool = False) -> pl.DataFrame:
    """Cached ce_logit of cross-encoder `fold` for the pruned candidates of a split.
    On train only the other fold and the validation queries are scored (never the model's own fold)."""
    path = WORK_DIR / (f"ce_{split}.parquet" if fold == "a" else f"ce_{split}_{fold}.parquet")
    if path.exists() and not force:
        return pl.read_parquet(path)
    cand = pl.read_parquet(WORK_DIR / f"pruned_{split}.parquet", columns=["q_row", "s1_row"])
    if split == "train":
        from .prune import valid_s1_rows
        va_q = cand.join(valid_s1_rows(), on="s1_row", how="semi").select("q_row").unique()
        other = cand.filter(fold_b() if fold == "a" else ~fold_b())
        cand = pl.concat([other, cand.join(va_q, on="q_row", how="semi")]).unique(maintain_order=True)
    cand = cand.with_columns(pl.Series("ce_logit", score_ce(cand, split, fold)))
    cand.write_parquet(path)
    return cand


def train_stage(fold: str = "a"):
    from .prune import add_labels, train_queries
    cand = pl.read_parquet(WORK_DIR / "pruned_train.parquet", columns=["q_row", "s1_row"])
    tr_q = pl.DataFrame({"q_row": train_queries(cand)}).filter(~fold_b() if fold == "a" else fold_b())
    pa = add_labels(cand.join(tr_q, on="q_row", how="semi"), "train").select("q_row", "s1_row", "label")
    pa = pa.sample(min(N_TRAIN_PAIRS, len(pa)), seed=SEED)
    print(f"  ce train pairs {len(pa):,} (pos {pa['label'].mean():.3f})", flush=True)
    train_ce(pa, CE_DIRS[fold])


if __name__ == "__main__":
    # python -m src.cross_encoder train {a|b} | score {a|b} train test
    if sys.argv[1] == "train":
        train_stage(sys.argv[2])
    else:
        for split in sys.argv[3:]:
            ce_scores(split, sys.argv[2], force=True)
