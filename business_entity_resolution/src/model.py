"""LightGBM pair classifier, per-query decision rule and threshold tuning for macro F0.5.

Training uses train-split S1 entities (hash split in config.is_valid_expr); validation scores
the held-out S1 entities with the exact competition metric.
"""
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from .config import SEED, WORK_DIR
from .metric import macro_f05

NON_FEATURES = {"q_row", "s1_row", "s1_id", "s23_id", "label", "is_valid", "country", "p"}

PARAMS = dict(
    objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1, seed=SEED, num_threads=14,
)


def feature_cols(df: pl.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURES and df[c].dtype.is_numeric()]


def train(df: pl.DataFrame, feats: list, rounds: int = 600, valid: pl.DataFrame = None, params=None):
    p = {**PARAMS, **(params or {})}
    dtr = lgb.Dataset(df.select(feats).to_numpy().astype(np.float32, copy=False), df["label"].to_numpy(), feature_name=feats, free_raw_data=True)
    sets, names = [dtr], ["train"]
    if valid is not None:
        sets.append(lgb.Dataset(valid.select(feats).to_numpy().astype(np.float32, copy=False), valid["label"].to_numpy(), reference=dtr))
        names.append("valid")
    return lgb.train(p, dtr, rounds, valid_sets=sets, valid_names=names,
                     callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50, verbose=False)] if valid is not None
                     else [lgb.log_evaluation(100)])


def predict(model, df: pl.DataFrame, feats: list, chunk: int = 5_000_000) -> np.ndarray:
    out = np.empty(len(df), dtype=np.float32)
    for i in range(0, len(df), chunk):
        out[i:i + chunk] = model.predict(df.slice(i, chunk).select(feats).to_numpy(), num_threads=14)
    return out


def assign(df: pl.DataFrame, t: float) -> pl.DataFrame:
    """Each query -> its best-scoring S1 if p >= t (each S2/S3 matches at most one S1)."""
    best = df.filter(pl.col("p") >= t).sort("p", descending=True).unique("s23_id", keep="first")
    return best.select("s1_id", "s23_id", "p")


def tune_threshold(df: pl.DataFrame, s1_ids, gt: pl.DataFrame, grid=None) -> tuple:
    grid = grid if grid is not None else np.round(np.arange(0.2, 0.96, 0.05), 2)
    top = df.sort("p", descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id", "p")
    res = []
    for t in grid:
        m = macro_f05(s1_ids, top.filter(pl.col("p") >= t), gt)
        res.append((float(t), m["f05"], m["precision_micro"], m["recall_micro"], m["singleton_acc"]))
        print(f"  t={t:.2f} f05={m['f05']:.4f} P={m['precision_micro']:.4f} R={m['recall_micro']:.4f} "
              f"single={m['singleton_acc']:.4f}", flush=True)
    best = max(res, key=lambda r: r[1])
    return best[0], res


def save_model(model, feats: list, t: float, name: str):
    model.save_model(str(WORK_DIR / f"{name}.lgb"))
    (WORK_DIR / f"{name}.json").write_text(json.dumps({"features": feats, "threshold": t}))


def load_model(name: str):
    meta = json.loads((WORK_DIR / f"{name}.json").read_text())
    return lgb.Booster(model_file=str(WORK_DIR / f"{name}.lgb")), meta["features"], meta["threshold"]
