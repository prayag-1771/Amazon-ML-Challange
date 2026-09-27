"""Simulate France's missing-region queries on validation: blank state_eq for a fraction of queries, re-predict."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, polars as pl, lightgbm as lgb
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
v = pl.read_parquet(W + "valid_feats.parquet")
ms = {k: (lgb.Booster(model_file=W + f"lgb_{k}.lgb"), json.load(open(W + f"lgb_{k}.json"))["features"]) for k in ("v5cf", "v6k")}
def score(d, t=0.75):
    p = sum(m.predict(d.select(f).to_numpy(), num_threads=14) for m, f in ms.values()) / 2
    d = d.select("s1_id", "s23_id").with_columns(pl.Series("p", p))
    top = d.sort("p", descending=True).unique("s23_id", keep="first")
    out = {}
    for t in (0.7, 0.75, 0.8):
        r = macro_f05(s1v["s1_id"], top.filter(pl.col("p") >= t).select("s1_id", "s23_id"), gt, by=s1v)
        out[t] = (round(r["f05"], 5), round(r["f05_US"], 5), round(r["f05_India"], 5), round(r["precision_micro"], 5), round(r["recall_micro"], 5))
    return out
print("base", score(v), flush=True)
for frac in (0.35, 1.0):
    hit = (pl.col("s23_id").hash(11) % 100) < int(frac * 100)
    print(f"state missing {frac}", score(v.with_columns(pl.when(hit).then(-1.0).otherwise(pl.col("state_eq")).alias("state_eq"))), flush=True)
