"""Cheap prune of the blocking union down to the final candidate set (= candidate_pairs.tsv).

A small LightGBM on blocking similarities + three fast rapidfuzz scores ranks the S1 candidates
of each query; the top N_KEEP per query (with p >= P_MIN) are kept for the full model.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .blocking import build_candidates, with_ids
from .config import SEED, WORK_DIR, is_valid_expr
from .features import blocking_feats, context_features
from .io_utils import load_ground_truth
from .normalize import normalize_source

N_KEEP = 6
P_MIN = 0.002
PRUNE_FEATS = ["tf_sim", "tf_rank", "emb_sim", "emb_rank", "emb_gap_best", "tf_gap_best",
               "nc_tset", "nc_ratio", "ac_tset", "q_ncand0"]


def _texts(split: str):
    s1 = normalize_source(split, 1).select("name_core", "addr_core")
    q = pl.concat([normalize_source(split, s).select("name_core", "addr_core") for s in (2, 3)])
    return s1, q


def cheap_features(cand: pl.DataFrame, split: str, chunk: int = 20_000_000) -> pl.DataFrame:
    s1, q = _texts(split)
    cand = blocking_feats(cand).with_columns(pl.len().over("q_row").cast(pl.Float32).alias("q_ncand0"))
    cols = {"nc_tset": [], "nc_ratio": [], "ac_tset": []}
    for i in range(0, len(cand), chunk):
        c = cand.slice(i, chunk)
        a, b = s1[c["s1_row"].to_numpy()], q[c["q_row"].to_numpy()]
        n1, n2 = a["name_core"].to_list(), b["name_core"].to_list()
        cols["nc_tset"].append(cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32))
        cols["nc_ratio"].append(cpdist(n1, n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32))
        cols["ac_tset"].append(cpdist(a["addr_core"].to_list(), b["addr_core"].to_list(),
                                      scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32))
    return cand.with_columns(**{k: pl.Series(k, np.concatenate(v)) for k, v in cols.items()})


def add_labels(cand: pl.DataFrame, split: str) -> pl.DataFrame:
    cand = with_ids(cand, split)
    gt = load_ground_truth().with_columns(pl.lit(1, pl.Int8).alias("label"))
    return cand.join(gt, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("label").fill_null(0))


def valid_s1_rows() -> pl.DataFrame:
    s1 = normalize_source("train", 1).select(pl.col("entity_id").alias("s1_id")).with_row_index("s1_row")
    return s1.filter(is_valid_expr()).select(pl.col("s1_row").cast(pl.UInt32))


def train_queries(cand: pl.DataFrame) -> pl.Series:
    """Queries none of whose candidates is a validation S1."""
    touch = cand.join(valid_s1_rows(), on="s1_row", how="semi")["q_row"].unique()
    return cand["q_row"].unique().filter(~cand["q_row"].unique().is_in(touch.implode()))


def train_prune(n_queries: int = 1_500_000) -> lgb.Booster:
    path = WORK_DIR / "prune.lgb"
    if path.exists():
        return lgb.Booster(model_file=str(path))
    cand = build_candidates("train")
    # train only on queries whose candidates are all train-split S1 (keeps validation clean)
    tr_q = train_queries(cand)
    qs = tr_q.sample(min(n_queries, len(tr_q)), seed=SEED)
    tr = cand.join(pl.DataFrame({"q_row": qs}), on="q_row", how="semi")
    tr = add_labels(cheap_features(tr, "train"), "train")
    ds = lgb.Dataset(tr.select(PRUNE_FEATS).to_numpy(), tr["label"].to_numpy(), feature_name=PRUNE_FEATS)
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500, verbose=-1,
                       seed=SEED, num_threads=14), ds, 300)
    m.save_model(str(path))
    return m


def prune(split: str, force: bool = False) -> pl.DataFrame:
    """Final candidate set with prune score p0 and all blocking features."""
    path = WORK_DIR / f"pruned_{split}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    model = train_prune()
    t = time.time()
    cand = cheap_features(build_candidates(split), split)
    p0 = np.empty(len(cand), np.float32)
    for i in range(0, len(cand), 10_000_000):
        p0[i:i + 10_000_000] = model.predict(cand.slice(i, 10_000_000).select(PRUNE_FEATS).to_numpy(), num_threads=14)
    cand = cand.with_columns(pl.Series("p0", p0))
    cand = cand.filter(
        (pl.col("p0").rank("ordinal", descending=True).over("q_row") <= N_KEEP) & (pl.col("p0") >= P_MIN)
    )
    # context over the *full* pruned set (per-S1 context must not depend on query sampling)
    cand = context_features(cand, "p0")
    cand = context_features(cand.drop("q_ncand", "s1_nq"), "nc_tset")
    cand = with_ids(cand, split)
    cand.write_parquet(path)
    print(f"  prune [{split}]: {len(cand):,} pairs kept {time.time() - t:.0f}s", flush=True)
    return cand


if __name__ == "__main__":
    for split in sys.argv[1:]:
        prune(split, force=True)
