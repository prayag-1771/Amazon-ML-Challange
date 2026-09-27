"""Validation (lgb_v5cf): how much macro F0.5 is recoverable by fixing queries in each band of top-1 p
(oracle decision inside the band, among existing candidates). Tells whether a heavier reranker on the
uncertain band is worth it."""
import sys
import polars as pl

sys.path.insert(0, ".")
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
from src.model import load_model, predict
from src.normalize import normalize_source

W = "../work/"
model, feats, t = load_model("lgb_v5cf")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats))).select("s1_id", "s23_id", "p", "label")
gt = load_ground_truth()
s1v = normalize_source("train", 1).filter(is_valid_expr("entity_id")).select(pl.col("entity_id").alias("s1_id"), "country")
ids = s1v["s1_id"]

top = va.sort("p", descending=True).unique("s23_id", keep="first")
base = top.filter(pl.col("p") >= t)
print("base", macro_f05(ids, base, gt)["f05"])
# oracle per query: assign the true S1 if it is among the candidates, else nothing
tru = va.filter(pl.col("label") == 1).unique("s23_id").select("s23_id", pl.col("s1_id").alias("s1_true"))
top = top.join(tru, on="s23_id", how="left")
edges = [0.0, 0.01, 0.05, 0.2, 0.5, 0.75, 0.9, 0.99, 0.999, 1.01]
for lo, hi in zip(edges[:-1], edges[1:]):
    band = (pl.col("p") >= lo) & (pl.col("p") < hi)
    fixed = pl.concat([
        top.filter(~band & (pl.col("p") >= t)).select("s1_id", "s23_id"),
        top.filter(band & pl.col("s1_true").is_not_null()).select(pl.col("s1_true").alias("s1_id"), "s23_id"),
    ])
    n = top.filter(band).height
    err = top.filter(band).with_columns(
        ((pl.col("p") >= t) & (pl.col("s1_id") != pl.col("s1_true").fill_null(""))).alias("fp"),
        ((pl.col("p") < t) & pl.col("s1_true").is_not_null()).alias("fn"),
        ((pl.col("p") >= t) & pl.col("s1_true").is_not_null() & (pl.col("s1_id") != pl.col("s1_true"))).alias("wrong_s1"))
    print(f"[{lo:.3f},{hi:.3f}) queries {n:>8,}  fp {err['fp'].sum():>6,}  fn {err['fn'].sum():>6,}  "
          f"wrong_s1 {err['wrong_s1'].sum():>5,}  oracle f05 {macro_f05(ids, fixed, gt)['f05']:.5f}")
