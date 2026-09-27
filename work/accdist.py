import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W = "../work/"
T = {"US": 0.75, "India": 0.80, "France": 0.75}
def acc(sc, split, filt=None):
    c = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    if filt is not None: c = c.filter(filt)
    d = pl.read_parquet(W + sc)
    top = d.sort("p2", descending=True).unique("s23_id", keep="first").join(c, on="s1_id")
    a = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T))
    n = c.join(a.group_by("s1_id").len("k"), on="s1_id", how="left").with_columns(pl.col("k").fill_null(0))
    return n, top
nv, tv = acc("valid_scores_stage2.parquet", "train", is_valid_expr("s1_id"))
nt, tt = acc("test_scores_stage2_v10.parquet", "test")
gt = load_ground_truth()
gv = nv.join(gt.group_by("s1_id").len("g"), on="s1_id", how="left").with_columns(pl.col("g").fill_null(0))
print("val accepted/S1, gold/S1:", gv.group_by("country").agg(pl.col("k").mean(), pl.col("g").mean()).rows())
print("test accepted/S1:", nt.group_by("country").agg(pl.col("k").mean()).rows())
for nm, n in (("val", nv), ("test", nt)):
    h = n.group_by("country", pl.col("k").clip(0, 8)).len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4)).sort("country", "k")
    print(nm, h.rows())
print("val gold dist", gv.group_by("country", pl.col("g").clip(0, 8)).len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4)).sort("country", "g").rows())
# score distribution of top-1 per query
for nm, t in (("val", tv), ("test", tt)):
    print(nm, t.group_by("country").agg([(pl.col("p2").is_between(a, b, closed="left")).mean().round(4).alias(f"{a}-{b}") for a, b in ((0.05, 0.3), (0.3, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 0.99), (0.99, 1.01))]).rows())
