import polars as pl
W = "../work/"
t = pl.read_parquet(W + "test_scores_stage2_v10.parquet")
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "business_name", "business_address", "country"]).rename({"entity_id": "s1_id", "business_name": "n1", "business_address": "a1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id", "business_name", "business_address", "country"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "business_name": "nq", "business_address": "aq", "country": "cq"})
top = t.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
print(top.group_by("country", "cq").len().sort("len"))
pl.Config.set_tbl_rows(200); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
for lo, hi in ((0.3, 0.6), (0.6, 0.75), (0.75, 0.9)):
    x = top.filter((pl.col("country") == "France") & (pl.col("p2") >= lo) & (pl.col("p2") < hi)).sample(25, seed=1)
    print(lo, hi); print(x.select(pl.col("p2").round(2), "n1", "nq", "a1", "aq"))
