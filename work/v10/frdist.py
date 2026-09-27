import sys
sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W = "../work/"
# test: per-country queries per S1 (by top-1 S1 of each query) and accepted per S1 per source distribution
d = pl.read_parquet(W + "test_scores_stage2_v9.parquet")
c = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
T = {"US": 0.75, "India": 0.80, "France": 0.75}
top = d.sort("p2", descending=True).unique("s23_id", keep="first").join(c, on="s1_id").with_columns(pl.col("s23_id").str.slice(0, 2).alias("src"))
acc = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T))
nq = {}
for s in (2, 3):
    q = pl.read_parquet(W + f"norm_test_s{s}.parquet", columns=["country"])["country"].value_counts()
    print("test source", s, q.sort("country").rows())
print("test S1", c["country"].value_counts().sort("country").rows())
cnt = c.join(acc.group_by("s1_id", "src").len().pivot(on="src", index="s1_id", values="len"), on="s1_id", how="left").fill_null(0)
for ctry in ("US", "India", "France"):
    x = cnt.filter(pl.col("country") == ctry)
    print(ctry, "pred per S1 S2 %.3f S3 %.3f" % (x["S2"].mean(), x["S3"].mean()),
          "dist S2", x["S2"].clip(0, 5).value_counts(normalize=True).sort("S2").select(pl.col("proportion").round(3)).to_series().to_list(),
          "dist S3", x["S3"].clip(0, 5).value_counts(normalize=True).sort("S3").select(pl.col("proportion").round(3)).to_series().to_list())
# val truth distribution
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth().with_columns(pl.col("s23_id").str.slice(0, 2).alias("src"))
g = s1v.join(gt.group_by("s1_id", "src").len().pivot(on="src", index="s1_id", values="len"), on="s1_id", how="left").fill_null(0)
for ctry in ("US", "India"):
    x = g.filter(pl.col("country") == ctry)
    print(ctry, "VAL TRUE per S1 S2 %.3f S3 %.3f" % (x["S2"].mean(), x["S3"].mean()),
          "dist S2", x["S2"].clip(0, 5).value_counts(normalize=True).sort("S2").select(pl.col("proportion").round(3)).to_series().to_list(),
          "dist S3", x["S3"].clip(0, 5).value_counts(normalize=True).sort("S3").select(pl.col("proportion").round(3)).to_series().to_list())
for s in (2, 3):
    q = pl.read_parquet(W + f"norm_train_s{s}.parquet", columns=["country"])["country"].value_counts()
    print("train source", s, q.sort("country").rows())
print("train S1", pl.read_parquet(W + "norm_train_s1.parquet", columns=["country"])["country"].value_counts().sort("country").rows())
