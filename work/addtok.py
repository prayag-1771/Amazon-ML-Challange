"""Token-set differences between S1 name_core and query name_core for top-1 pairs with the same address number.
Validation (labeled, US/India) vs test France (p2 only)."""
import polars as pl
pl.Config.set_tbl_rows(70); pl.Config.set_tbl_width_chars(250)
W = "../work/"
def load(split, scores, extra=[]):
    d = pl.read_parquet(W + scores)
    d = d.sort("p2", descending=True).unique("s23_id", keep="first")
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "name_core", "country", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    d = d.join(s1, on="s1_id").join(q, on="s23_id")
    d = d.with_columns(pl.col("n1").str.split(" ").alias("t1"), pl.col("nq").str.split(" ").alias("tq"))
    d = d.with_columns(pl.col("tq").list.set_difference("t1").alias("add"), pl.col("t1").list.set_difference("tq").alias("drop"))
    return d.with_columns((pl.col("u1").str.split(" ").list.first() == pl.col("uq").str.split(" ").list.first()).fill_null(False).alias("num_eq"))
v = load("train", "valid_scores_stage2.parquet")
t = load("test", "test_scores_stage2_v9.parquet").filter(pl.col("country") == "France")
for nm, d in [("VAL", v), ("FR", t)]:
    one = d.filter(pl.col("num_eq") & (pl.col("add").list.len() == 1) & (pl.col("drop").list.len() == 0)).with_columns(pl.col("add").list.first().alias("tok"))
    aggs = [pl.len(), pl.col("p2").mean().round(3).alias("p2"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")]
    if nm == "VAL": aggs.append(pl.col("label").mean().round(3).alias("lab"))
    print(nm, "added-one-token pairs", len(one))
    print(one.group_by("tok").agg(aggs).sort("len", descending=True).head(45))
