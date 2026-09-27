"""Simulate test distractor density on validation: replicate distractor queries (no true S1) by weight w, retune t."""
import sys
sys.path.insert(0, ".")
import numpy as np
import polars as pl
from src.metric import macro_f05
from src.config import is_valid_expr
W = "../work/"
gt = pl.read_parquet(W + "gt_pairs.parquet")
va = pl.read_parquet(W + "valid_scores_lgb_v5cf.parquet")
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
top = va.sort("p", descending=True).unique("s23_id", keep="first").join(s1v, on="s1_id", how="semi")
dis = top.join(gt, on="s23_id", how="anti")  # queries that match nothing
print("valid top rows", top.height, "distractor queries", dis.height, "distractors with p>=.02", dis.filter(pl.col("p") >= 0.02).height)
rng = np.random.default_rng(0)
grid = [0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.94, 0.96]
for w in (1.0, 1.4, 1.7, 2.0):
    extra = []
    k = w - 1
    rep = 0
    while k > 1e-9:
        frac = min(k, 1.0)
        d = dis.filter(pl.Series(rng.random(dis.height) < frac)).with_columns((pl.col("s23_id") + f"_dup{rep}").alias("s23_id"))
        extra.append(d); k -= frac; rep += 1
    sim = pl.concat([top] + extra) if extra else top
    res = []
    for t in grid:
        m = macro_f05(s1v["s1_id"], sim.filter(pl.col("p") >= t), gt, by=s1v)
        res.append((t, round(m["f05"], 5), round(m["f05_US"], 5), round(m["f05_India"], 5), round(m["precision_micro"], 5), round(m["recall_micro"], 5)))
    print("w", w)
    for r in res: print("   ", r)
