"""Hard-pair cross-encoder (v10): a cross-encoder fine-tuned on the ambiguous band of training pairs
(out-of-fold |ce_logit| < LIM from the cross-fitted e5-small pair) plus a sample of easy pairs.

It is used only as a stage-2 feature, so it never needs out-of-fold scores on train: validation S1 entities
and every query touching them are excluded from its training data, and it is scored on the uncertain band
(first-stage ensemble p in [P_LO, P_HI]) of validation and test pairs.

  python -m src.ce_hard train <tag> <init> <epochs> <lr>   # init: HF id / local dir (e.g. work/ce_e5s)
  python -m src.ce_hard score <tag>
"""
import sys

import numpy as np
import polars as pl

from . import cross_encoder as ce
from .config import SEED, WORK_DIR
from .prune import add_labels, valid_s1_rows

LIM, EASY_FRAC = 6.0, 0.25
P_LO, P_HI = 1e-3, 0.9999

def hard_pairs() -> pl.DataFrame:
    p = pl.read_parquet(WORK_DIR / "pruned_train.parquet", columns=["q_row", "s1_row"])
    va_q = p.join(valid_s1_rows(), on="s1_row", how="semi").select("q_row").unique()
    o = pl.concat([pl.read_parquet(WORK_DIR / "ce_train.parquet"), pl.read_parquet(WORK_DIR / "ce_train_b.parquet")])
    o = o.join(va_q, on="q_row", how="anti").unique(["q_row", "s1_row"])
    hard = o.filter(pl.col("ce_logit").abs() < LIM)
    easy = o.filter(pl.col("ce_logit").abs() >= LIM).sample(int(EASY_FRAC * len(hard)), seed=SEED)
    return add_labels(pl.concat([hard, easy]).select("q_row", "s1_row"), "train").select("q_row", "s1_row", "label")

def band(split: str) -> pl.DataFrame:
    """(q_row, s1_row) of the pairs whose first-stage ensemble score lies in the uncertain band."""
    nm = "valid" if split == "train" else "test"
    s = pl.read_parquet(WORK_DIR / f"{nm}_scores_lgb_v5cf.parquet", columns=["s1_id", "s23_id", "p"]).join(
        pl.read_parquet(WORK_DIR / f"{nm}_scores_lgb_v6k.parquet", columns=["s1_id", "s23_id", "p"]).rename({"p": "pb"}),
        on=["s1_id", "s23_id"])
    s = s.filter(((pl.col("p") + pl.col("pb")) / 2).is_between(P_LO, P_HI))
    f = pl.read_parquet(WORK_DIR / f"{nm}_feats.parquet", columns=["q_row", "s1_row", "s1_id", "s23_id"])
    return f.join(s.select("s1_id", "s23_id"), on=["s1_id", "s23_id"], how="semi").select("q_row", "s1_row")

if __name__ == "__main__":
    tag = sys.argv[2]
    ce.CE_DIRS[tag] = WORK_DIR / f"ce_hard_{tag}"
    if sys.argv[1] == "train":
        ce.BASE = sys.argv[3]
        pa = hard_pairs()
        print(f"  hard-pair ce train pairs {len(pa):,} (pos {pa['label'].mean():.3f})", flush=True)
        ce.train_ce(pa, ce.CE_DIRS[tag], epochs=int(sys.argv[4]), lr=float(sys.argv[5]))
    else:
        ce.BASE = str(ce.CE_DIRS[tag])
        for split in ("train", "test"):
            b = band(split)
            b = b.with_columns(pl.Series("ce_h", ce.score_ce(b, split, tag)))
            b.write_parquet(WORK_DIR / f"ce_hard_{tag}_{'valid' if split == 'train' else 'test'}.parquet")
