import sys; sys.path.insert(0, ".")
import polars as pl
W = "../work/"
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(260)
t = pl.read_parquet(W + "test_scores_stage2_v10.parquet")
top = t.sort("p2", descending=True).unique("s23_id", keep="first")
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country", "name_core", "nums", "business_name", "business_address"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1", "business_name": "bn1", "business_address": "ba1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id", "name_core", "nums", "business_name", "business_address"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq", "business_name": "bnq", "business_address": "baq"})
x = top.join(s1, on="s1_id").join(q, on="s23_id")
a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False); b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
x = x.with_columns((b - a).alias("df"))
f = x.filter((pl.col("country") == "France") & (pl.col("n1") == pl.col("nq")) & (pl.col("df") < -1) & (pl.col("p2") < 0.75))
print(len(f)); print(f.sample(40, seed=5).select(pl.col("p2").round(2), "bn1", "bnq", "ba1", "baq"))
# how many other candidates in the S1 have the same exact name?
