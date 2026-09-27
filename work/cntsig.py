"""Label-free token verdict: for top-1 pairs whose query adds one token, count OTHER confident (p2>=0.9) queries
on the same S1. True-match tokens -> S1 already 'uses up' this query -> ~1 fewer other matches than distractor tokens."""
import sys; sys.path.insert(0, "../work")
import polars as pl
from addtok import load
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
def run(d, lab):
    d = d.with_columns(((pl.col("p2") >= 0.9).cast(pl.Int32).sum().over("s1_id")).alias("s1_conf"))
    d = d.with_columns((pl.col("s1_conf") - (pl.col("p2") >= 0.9).cast(pl.Int32)).alias("others"))
    base = d.filter(pl.col("p2") >= 0.9)["others"].mean()
    one = d.filter((pl.col("add").list.len() == 1) & (pl.col("drop").list.len() == 0) & (pl.col("p2") > 0.01)).with_columns(pl.col("add").list.first().alias("tok"))
    aggs = [pl.len(), pl.col("p2").mean().round(2).alias("p2"), pl.col("others").mean().round(2).alias("others")]
    if lab: aggs.append(pl.col("label").mean().round(2).alias("lab"))
    print("baseline others (conf pairs):", round(base, 3))
    return one.group_by("country", "tok").agg(aggs).filter(pl.col("len") >= 40).sort("country", "len", descending=[False, True])
print(run(load("train", "valid_scores_stage2.parquet"), True).group_by("country", maintain_order=True).head(25))
print(run(load("test", "test_scores_stage2_v9.parquet"), False).group_by("country", maintain_order=True).head(40))
