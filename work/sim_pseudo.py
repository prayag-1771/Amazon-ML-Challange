"""Does pseudo-labelling an unlabelled country help? Simulation with India playing France:
  A) LightGBM trained on US labels only, evaluated on India validation S1.
  B) A + India pairs pseudo-labelled by A's own confident predictions (no India labels used).
  C) Upper bound: US + India true labels (same query sample).
Uses the same 64 features as lgb_v5cf (cross-fitted cross-encoder)."""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, ".")
import src.run_pipeline as rp
from src.config import SEED, WORK_DIR, is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
from src.model import PARAMS
from src.normalize import normalize_source
from src.prune import prune, train_queries

rp.CROSS_FIT = True
NQ = 600_000  # train queries per country
feats = json.loads((WORK_DIR / "lgb_v5cf.json").read_text())["features"]
gt = load_ground_truth()
s1 = normalize_source("train", 1).select(pl.col("entity_id").alias("s1_id"), "country")
cache = WORK_DIR / "sim_train_feats.parquet"
if cache.exists():
    tr = pl.read_parquet(cache)
else:
    cand = prune("train")
    s1r = normalize_source("train", 1).select("country").with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
    qc = cand.select("q_row", "s1_row").unique("q_row").join(s1r, on="s1_row").select("q_row", "country")
    qc = qc.join(pl.DataFrame({"q_row": train_queries(cand)}), on="q_row", how="semi")
    qs = pl.concat([qc.filter(pl.col("country") == c).sample(NQ, seed=SEED) for c in ("US", "India")])
    tr = rp.add_ce(rp.featurize(cand.join(qs.select("q_row"), on="q_row", how="semi"), "train"), "train")
    tr = tr.join(gt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "s23_id"], how="left")
    tr = tr.with_columns(pl.col("label").fill_null(0)).join(s1, on="s1_id")
    tr.write_parquet(cache)
va = pl.read_parquet(WORK_DIR / "valid_feats.parquet").join(s1, on="s1_id")
print(f"train {len(tr):,} pairs, valid {len(va):,}", flush=True)
s1v = s1.filter(is_valid_expr("s1_id"))
ids = {c: s1v.filter(pl.col("country") == c)["s1_id"] for c in ("US", "India")}


def X(d):
    return d.select(feats).to_numpy().astype(np.float32, copy=False)


def fit(d, w=None, name=""):
    t = time.time()
    ds = lgb.Dataset(X(d), d["label"].to_numpy(), weight=w, feature_name=feats, free_raw_data=True)
    us_va = va.filter(pl.col("country") == "US")
    dv = lgb.Dataset(X(us_va), us_va["label"].to_numpy(), reference=ds)
    m = lgb.train(PARAMS, ds, 1500, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"  [{name}] trained {m.best_iteration} rounds {time.time() - t:.0f}s", flush=True)
    return m


def score(m, name):
    d = va.with_columns(pl.Series("p", m.predict(X(va), num_threads=14)))
    top = d.sort("p", descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id", "p")
    out = []
    for c in ("US", "India"):
        row = [f"{c}:"]
        for t in (0.5, 0.65, 0.75, 0.85):
            row.append(f"t{t}={macro_f05(ids[c], top.filter(pl.col('p') >= t), gt)['f05']:.5f}")
        out.append(" ".join(row))
    print(f"{name}: " + " | ".join(out), flush=True)
    return d


us = tr.filter(pl.col("country") == "US")
mA = fit(us, name="A us-only")
score(mA, "A us-only")

# pseudo-label India: unlabelled pool = India train queries + India validation queries (transductive, like France test)
pool = pl.concat([tr.filter(pl.col("country") == "India").drop("label"),
                  va.filter(pl.col("country") == "India").drop("label")], how="diagonal_relaxed")
pool = pool.with_columns(pl.Series("p", mA.predict(X(pool), num_threads=14)))
pool = pool.with_columns(pl.col("p").rank("ordinal", descending=True).over("q_row").alias("r"),
                         pl.col("p").max().over("q_row").alias("p1"),
                         pl.col("p").top_k(2).min().over("q_row").alias("p2"),
                         pl.len().over("q_row").alias("nc"))
pool = pool.with_columns(pl.when(pl.col("nc") == 1).then(0.0).otherwise(pl.col("p2")).alias("p2"))
for hi, lo in ((0.99, 0.01), (0.95, 0.05)):
    sure_pos = (pl.col("p1") >= hi) & (pl.col("p2") <= lo)
    sure_neg = pl.col("p1") <= lo
    ps = pool.filter(sure_pos | sure_neg).with_columns(
        pl.when(sure_pos & (pl.col("r") == 1)).then(1).otherwise(0).cast(pl.Int8).alias("label"))
    print(f"pseudo hi={hi}: {ps['q_row'].n_unique():,}/{pool['q_row'].n_unique():,} India queries, "
          f"{len(ps):,} pairs, pos {ps['label'].mean():.3f}", flush=True)
    d = pl.concat([us, ps.select(us.columns)])
    score(fit(d, name=f"B pseudo {hi}"), f"B pseudo {hi}")
    del d, ps

score(fit(tr, name="C us+india true"), "C us+india true")
