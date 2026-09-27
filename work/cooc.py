"""For top-1 pairs where the query adds exactly one token vs the S1 name: how many OTHER queries whose top-1 is the
same S1 add the same token (same source / other source). Distractor entities may form their own record clusters."""
import sys; sys.path.insert(0, "../work")
import polars as pl
from addtok import load
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
v = load("train", "valid_scores_stage2.parquet")
t = load("test", "test_scores_stage2_v9.parquet").filter(pl.col("country") == "France")
def co(d):
    d = d.filter(pl.col("p2") > 0.02).with_columns(pl.col("s23_id").str.slice(0, 2).alias("src"),
        pl.when((pl.col("add").list.len() == 1) & (pl.col("drop").list.len() == 0)).then(pl.col("add").list.first()).alias("tok"))
    c = d.filter(pl.col("tok").is_not_null()).group_by("s1_id", "tok", "src").agg(pl.len().alias("n"))
    c2 = c.filter(pl.col("src") == "S2").select("s1_id", "tok", pl.col("n").alias("n2"))
    c3 = c.filter(pl.col("src") == "S3").select("s1_id", "tok", pl.col("n").alias("n3"))
    d = d.join(c2, on=["s1_id", "tok"], how="left").join(c3, on=["s1_id", "tok"], how="left").with_columns(pl.col("n2", "n3").fill_null(0))
    return d.filter(pl.col("tok").is_not_null()).with_columns(
        pl.when(pl.col("src") == "S2").then(pl.col("n3")).otherwise(pl.col("n2")).alias("other"),
        (pl.when(pl.col("src") == "S2").then(pl.col("n2")).otherwise(pl.col("n3")) - 1).alias("same"))
vv, tt = co(v), co(t)
print(vv.group_by("tok").agg(pl.len(), pl.col("label").mean().round(3).alias("lab"), (pl.col("other") > 0).mean().round(3).alias("oth>0"), (pl.col("same") > 0).mean().round(3).alias("same>0")).sort("len", descending=True).head(25))
print(tt.group_by("tok").agg(pl.len(), pl.col("p2").mean().round(3).alias("p2"), (pl.col("other") > 0).mean().round(3).alias("oth>0"), (pl.col("same") > 0).mean().round(3).alias("same>0")).sort("len", descending=True).head(30))
print(vv.group_by("label", (pl.col("other") > 0).alias("o")).len().sort("label", "o"))
