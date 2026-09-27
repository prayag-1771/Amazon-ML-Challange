import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W = "../work/"
T = {"US": 0.75, "India": 0.80, "France": 0.75}
t = pl.read_parquet(W + "test_scores_stage2_v10.parquet")
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id", "country"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "country": "qc"})
top = t.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id")
acc = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T))
print("queries/S1:", q.group_by("qc").len().join(s1.group_by("country").len().rename({"country": "qc", "len": "ns1"}), on="qc").with_columns((pl.col("len") / pl.col("ns1")).round(3)).to_dicts())
c = s1.join(acc.group_by("s1_id").len("k"), on="s1_id", how="left").with_columns(pl.col("k").fill_null(0))
print(c.group_by("country").agg(pl.col("k").mean().round(3).alias("acc_per_s1"), (pl.col("k") == 0).mean().round(4).alias("zero")).sort("country"))
print(c.group_by("country", pl.col("k").clip(0, 8)).len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4)).pivot(on="k", index="country", values="len", sort_columns=True))
# val
gt = pl.read_parquet(W + "gt_pairs.parquet")
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
v = pl.read_parquet(W + "valid_scores_stage2.parquet")
tv = v.sort("p2", descending=True).unique("s23_id", keep="first").join(s1v, on="s1_id")
av = tv.filter(pl.col("p2") >= pl.col("country").replace_strict(T))
cv = s1v.join(av.group_by("s1_id").len("k"), on="s1_id", how="left").join(gt.group_by("s1_id").len("nt"), on="s1_id", how="left").fill_null(0)
print(cv.group_by("country").agg(pl.col("k").mean().round(3).alias("acc_per_s1"), pl.col("nt").mean().round(3).alias("true_per_s1"), (pl.col("k") == 0).mean().round(4).alias("zero"), (pl.col("nt") == 0).mean().round(4).alias("true_zero")))
print(cv.group_by("country", pl.col("k").clip(0, 8)).len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4)).pivot(on="k", index="country", values="len", sort_columns=True))
