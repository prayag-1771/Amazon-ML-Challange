"""Simulate test-like distractor density on validation: drop a fraction of S1 entities (their queries become unmatched)."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
d = pl.read_parquet(W + "valid_scores_stage2.parquet")
bins = [0, 0.01, 0.05, 0.2, 0.5, 0.75, 0.9, 0.97, 0.99, 1.01]
for f in (0.0, 0.2, 0.3, 0.4, 0.5):
    keep = s1v.filter((pl.col("s1_id").hash(11) % 1000) >= int(f * 1000))
    dd = d.join(pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id"]).rename({"entity_id": "s1_id"})
                .join(s1v.join(keep, on="s1_id", how="anti"), on="s1_id", how="anti"), on="s1_id", how="semi")
    top = dd.sort("p2", descending=True).unique("s23_id", keep="first").join(keep, on="s1_id")
    h = np.histogram(top.filter(pl.col("country") == "US")["p2"], bins)[0] / keep.filter(pl.col("country") == "US").height
    res = []
    for t in (0.75, 0.8, 0.85, 0.9, 0.93):
        m = macro_f05(keep["s1_id"], top.filter(pl.col("p2") >= t), gt, by=keep)
        res.append(f"t={t} {m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f})")
    print(f"f={f}", " | ".join(res)); print("   US top1 hist per S1", np.round(h, 3), flush=True)
