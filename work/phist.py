import sys
sys.path.insert(0, ".")
import numpy as np
import polars as pl
from src.model import load_model, predict

W = "../work/"
model, feats, T = load_model("lgb_v3")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats)))
s1 = pl.read_parquet(W + "raw_train_s1.parquet").select(pl.col("entity_id").alias("s1_id"), "country")
vtop = va.sort("p", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id")
# a validation query's top S1 may be non-held-out; keep anyway (it's the model's view)
sc = pl.read_parquet(W + "test_scores_lgb_v3.parquet")
s1t = pl.read_parquet(W + "raw_test_s1.parquet").select(pl.col("entity_id").alias("s1_id"), "country")
ttop = sc.sort("p", descending=True).unique("s23_id", keep="first").join(s1t, on="s1_id")
bins = [0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75, 0.9, 0.99, 1.01]
def hist(df, name):
    for c in sorted(df["country"].unique()):
        p = df.filter(pl.col("country") == c)["p"].to_numpy()
        h = np.histogram(p, bins)[0] / len(p)
        print(f"{name:5s} {c:7s} n={len(p):>9,} " + " ".join(f"{x:.4f}" for x in h))
print("bins", bins)
hist(vtop, "valid")
hist(ttop, "test")
# valid: label-based rates per bin
vt = vtop.with_columns(pl.col("label").cast(pl.Float64))
vt = vt.with_columns(pl.col("p").cut(bins[1:-1]).alias("b"))
print(vt.group_by("b").agg(pl.len(), pl.col("label").mean(), pl.col("p").mean()).sort("b"))
