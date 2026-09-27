"""Emulate test distractor density on validation: unmatched queries are 1.8x as frequent per S1 on test
(mixture fit of top-1 score histograms: US 1.78, India 1.81). Duplicate 80% of unmatched validation queries."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
d = pl.read_parquet(W + "valid_scores_stage2.parquet")
top = d.sort("p2", descending=True).unique("s23_id", keep="first").join(s1v, on="s1_id")
um = top.join(gt, on="s23_id", how="anti")
def dup(top, frac, seed):
    x = um.filter((pl.col("s23_id").hash(seed) % 1000) < int(frac * 1000)).with_columns((pl.col("s23_id") + "_d").alias("s23_id"))
    return pl.concat([top, x])
for frac in (0.0, 0.8):
    tt = dup(top, frac, 5)
    for t in (0.7, 0.75, 0.8, 0.85, 0.9):
        m = macro_f05(s1v["s1_id"], tt.filter(pl.col("p2") >= t), gt, by=s1v)
        print(f"dup={frac} t={t} {m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f}) prec {m['precision_micro']:.5f}", flush=True)
