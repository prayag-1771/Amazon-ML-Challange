import polars as pl
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(250)
W = "../work/"
src = open(W + "numrel2.py", encoding="utf-8").read().split("one = pl.read_parquet")[0]
exec(src)

v = pl.read_parquet(W + "valid_scores_stage2.parquet").sort("p2", descending=True).unique("s23_id", keep="first")
v = rel(v, "train").join(pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id")
t = pl.read_parquet(W + "test_scores_stage2_v9.parquet").sort("p2", descending=True).unique("s23_id", keep="first")
t = rel(t, "test").join(pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id")
b = pl.col("p2").cut([0.05, 0.3, 0.75, 0.95])
print("VAL", v.group_by("country", "nrel").agg(pl.len(), pl.col("label").mean().round(3).alias("lab"), pl.col("p2").mean().round(3).alias("p2"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc"),
      ((pl.col("p2") > 0.05) & (pl.col("p2") < 0.95)).mean().round(3).alias("mid")).sort("country", "nrel"))
print("TEST", t.group_by("country", "nrel").agg(pl.len(), pl.col("p2").mean().round(3).alias("p2"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc"),
      ((pl.col("p2") > 0.05) & (pl.col("p2") < 0.95)).mean().round(3).alias("mid")).sort("country", "nrel"))
