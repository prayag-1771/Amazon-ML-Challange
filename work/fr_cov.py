"""Per country: share of S2/S3 test queries with any candidate, and name/address traits of those without."""
import polars as pl

W = "../work/"
cols = ["entity_id", "country", "business_name", "business_address", "addr_empty", "name_full"]
q = pl.concat([pl.read_parquet(W + f"norm_test_s{k}.parquet", columns=cols) for k in (2, 3)])
tr = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=cols) for k in (2, 3)])
c = pl.read_parquet(W + "pruned_test.parquet", columns=["s23_id"]).unique()
q = q.with_columns(pl.col("entity_id").is_in(c["s23_id"].implode()).alias("has_cand"),
                   pl.col("business_name").str.contains(r"[^\x00-\x7F]").alias("non_ascii"))
print(q.group_by("country").agg(pl.len(), pl.col("has_cand").mean(), pl.col("addr_empty").mean(),
                                pl.col("non_ascii").mean()).sort("country"))
print("train addr_empty:", tr.group_by("country").agg(pl.col("addr_empty").mean()))
pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250)
print(q.filter((pl.col("country") == "France") & ~pl.col("has_cand")).sample(25, seed=0)
       .select("business_name", "business_address"))
