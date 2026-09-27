"""Per (name-relation, number-relation) category of each query's top-1: share of queries, accept rate.
US/India from val (plus true rate), France from test. Label-free comparison of model behaviour."""
import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=200)
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W = "../work/"
exec(open("../work/v10/frcat.py").read().split("def load")[0].split("W = \"../work/\"")[1])
def load(split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    return s1, q
T = 0.75
s1, q = load("test")
d = pl.read_parquet(W + "test_scores_stage2_v9.parquet")
fr = cats(d.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id").filter(pl.col("country") == "France"))
nfr = s1.filter(pl.col("country") == "France").height
s1, q = load("train")
v = pl.read_parquet(W + "valid_scores_stage2.parquet")
vt = cats(v.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id"))
nv = s1.filter(is_valid_expr("s1_id")).group_by("country").len()
print(nv)
out = fr.group_by("nm", "num").agg((pl.len() / nfr).alias("FR_q"), ((pl.col("p2") >= T).sum() / nfr).alias("FR_acc"))
for c in ("US", "India"):
    n = nv.filter(pl.col("country") == c)["len"][0]
    x = vt.filter(pl.col("country") == c).group_by("nm", "num").agg(
        (pl.len() / n).alias(f"{c}_q"), ((pl.col("p2") >= T).sum() / n).alias(f"{c}_acc"),
        (pl.col("label").sum() / n).alias(f"{c}_true"), ((pl.col("label") == 1) & (pl.col("p2") >= T)).sum().truediv(n).alias(f"{c}_tp"))
    out = out.join(x, on=["nm", "num"], how="full", coalesce=True)
print(out.sort("FR_q", descending=True).with_columns(pl.selectors.float().round(4)))
gt = load_ground_truth()
s1t = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
m = s1t.join(gt.select("s1_id").unique().with_columns(pl.lit(1).alias("has")), on="s1_id", how="left")
print("train singleton rate", m.group_by("country").agg(pl.col("has").is_null().mean()))
