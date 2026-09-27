"""End-to-end pipeline.

  python -m src.run_pipeline --stage all
stages: normalize | embed | block | prune | ce | train | predict  (each caches under work/)

Validation: queries touching a held-out S1 (10% hash split) are scored with the exact
competition metric over the held-out S1 entities; training uses queries not touching them.
"""
import argparse
import time

import numpy as np
import polars as pl

from .blocking import block_emb, block_tfidf, build_candidates
from .config import OUTPUT_DIR, SEED, WORK_DIR, is_valid_expr
from .cross_encoder import CE_DIRS, ce_scores, fold_b, train_stage as train_ce_stage
from .embed import embed_source, load_model as load_embedder
from .features import load_side, pair_features
from .io_utils import load_ground_truth, load_source, write_id_lists
from .metric import macro_f05
from .model import assign, feature_cols, load_model, predict, save_model, train, tune_threshold
from .normalize import normalize_source
from .prune import prune, train_queries, valid_s1_rows
from .token_stats import token_features

MODEL_NAME = "lgb_v6k"
N_TRAIN_Q = 20_000_000  # i.e. all train queries (~8.5M)
USE_CE = True  # cross-encoder logit as a feature (src/cross_encoder.py)
CROSS_FIT = True  # two cross-encoders (fold A / fold B): LightGBM can then train on all queries
# lgb_v5cf (v6) predates the context-keyed token statistics (v7): it is trained without them (64 features)
MODEL_DROP = {"lgb_v5cf": {"xq_krelmin", "xq_krelmean", "xq_kfmin", "x1_krelmin", "x1_krelmean", "x1_kfmin"}}


def featurize(cand: pl.DataFrame, split: str, chunk: int = 5_000_000) -> pl.DataFrame:
    s1, q = load_side(split)
    parts = []
    for i in range(0, len(cand), chunk):
        t = time.time()
        parts.append(token_features(pair_features(cand.slice(i, chunk), s1, q), split, s1, q))
        print(f"  features {split}: {min(i + chunk, len(cand)):,}/{len(cand):,} {time.time() - t:.0f}s", flush=True)
    return pl.concat(parts)


def ce_logits(split: str) -> pl.DataFrame:
    """Out-of-fold cross-encoder logit per pair. Train: the logit of the model that did not see the
    query (fold A <-> fold B); validation and test: mean of both models (CROSS_FIT) or model A only."""
    a = ce_scores(split, "a")
    if not CROSS_FIT:
        return a
    b = ce_scores(split, "b").rename({"ce_logit": "ce_b"})
    d = a.join(b, on=["q_row", "s1_row"], how="full", coalesce=True)
    if split == "train":
        va_q = d.join(valid_s1_rows(), on="s1_row", how="semi").select("q_row").unique()
        is_va = pl.col("q_row").is_in(va_q["q_row"].implode())
        logit = (pl.when(is_va).then((pl.col("ce_logit") + pl.col("ce_b")) / 2)
                 .when(fold_b()).then(pl.col("ce_logit")).otherwise(pl.col("ce_b")))
    else:
        logit = (pl.col("ce_logit") + pl.col("ce_b")) / 2
    return d.select("q_row", "s1_row", logit.alias("ce_logit"))


def add_ce(df: pl.DataFrame, split: str) -> pl.DataFrame:
    """Cross-encoder logit + its within-query context (query-level only: without cross-fitting,
    fold-A train queries are never scored, so per-S1 context would be incomplete)."""
    ce = ce_logits(split)
    df = df.join(ce, on=["q_row", "s1_row"], how="left")
    assert df["ce_logit"].null_count() == 0, "pairs without ce_logit"
    return df.with_columns(
        pl.col("ce_logit").rank("ordinal", descending=True).over("q_row").cast(pl.Float32).alias("ce_qrank"),
        (pl.col("ce_logit") - pl.col("ce_logit").max().over("q_row")).alias("ce_gap_best"),
        (pl.col("ce_logit") - pl.col("ce_logit").top_k(2).min().over("q_row")).alias("ce_gap_2nd"),
    )


def stage_train():
    cand = prune("train")
    gt = load_ground_truth()
    vrows = valid_s1_rows()
    tr_q = train_queries(cand)
    if USE_CE and not CROSS_FIT:  # the cross-encoder was trained on fold A queries - LightGBM sees fold B only
        tr_q = pl.DataFrame({"q_row": tr_q}).filter(fold_b())["q_row"]
    tr_q = tr_q.sample(min(N_TRAIN_Q, len(tr_q)), seed=SEED)
    va_q = cand.join(vrows, on="s1_row", how="semi")["q_row"].unique()
    tr = featurize(cand.join(pl.DataFrame({"q_row": tr_q}), on="q_row", how="semi"), "train")
    va = featurize(cand.join(pl.DataFrame({"q_row": va_q}), on="q_row", how="semi"), "train")
    if USE_CE:
        tr, va = add_ce(tr, "train"), add_ce(va, "train")
    gl = gt.with_columns(pl.lit(1, pl.Int8).alias("label"))
    tr = tr.join(gl, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("label").fill_null(0))
    va = va.join(gl, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("label").fill_null(0))
    va.write_parquet(WORK_DIR / "valid_feats.parquet")
    feats = [f for f in feature_cols(tr) if f not in MODEL_DROP.get(MODEL_NAME, set())]
    print(f"train pairs {len(tr):,} (pos {tr['label'].mean():.3f}), valid pairs {len(va):,}, {len(feats)} feats")
    model = train(tr, feats, rounds=2000, valid=va)
    va = va.with_columns(pl.Series("p", predict(model, va, feats)))
    s1v = normalize_source("train", 1).filter(is_valid_expr("entity_id")).select(
        pl.col("entity_id").alias("s1_id"), "country")
    t, _ = tune_threshold(va, s1v["s1_id"], gt)
    m = macro_f05(s1v["s1_id"], assign(va, t), gt, by=s1v)
    print("VALID", t, m)
    imp = sorted(zip(feats, model.feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(f, round(g)) for f, g in imp[:20]])
    save_model(model, feats, t, MODEL_NAME)
    with open(WORK_DIR / "experiments.csv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{MODEL_NAME},{t},{m['f05']:.5f},{m.get('f05_US', 0):.5f},"
                f"{m.get('f05_India', 0):.5f},{len(feats)}\n")


def stage_predict():
    model, feats, t = load_model(MODEL_NAME)
    cand = prune("test")
    df = featurize(cand, "test")
    if USE_CE:
        df = add_ce(df, "test")
    df = df.with_columns(pl.Series("p", predict(model, df, feats)))
    df.select("s1_id", "s23_id", "p").write_parquet(WORK_DIR / f"test_scores_{MODEL_NAME}.parquet")
    s1_ids = load_source("test", 1)["entity_id"].to_list()
    matches = assign(df, t)
    write_id_lists(OUTPUT_DIR / "candidate_pairs.tsv", s1_ids, cand, "candidate_entity_ids")
    write_id_lists(OUTPUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    print(f"test: {len(cand):,} candidate pairs, {len(matches):,} matches, "
          f"{matches['s1_id'].n_unique():,}/{len(s1_ids):,} S1 with >=1 match")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["normalize", "embed", "block", "prune", "ce", "train", "predict", "all"])
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--cross-fit", action="store_true")
    a = ap.parse_args()
    globals()["MODEL_NAME"] = a.model
    globals()["CROSS_FIT"] = CROSS_FIT or a.cross_fit
    run = lambda s: a.stage in (s, "all")
    for split in ("train", "test"):
        if run("normalize"):
            for s in (1, 2, 3):
                normalize_source(split, s)
        if run("embed"):
            emb = load_embedder()
            for s in (1, 2, 3):
                embed_source(split, s, emb)
        if run("block"):
            block_tfidf(split)
            block_emb(split)
            build_candidates(split)
        if run("prune"):
            prune(split)
    if USE_CE and run("ce"):
        for fold in ("a", "b") if CROSS_FIT else ("a",):
            if not CE_DIRS[fold].exists():
                train_ce_stage(fold)
            for split in ("train", "test"):
                ce_scores(split, fold)
    if run("train"):
        stage_train()
    if run("predict"):
        stage_predict()


if __name__ == "__main__":
    main()
