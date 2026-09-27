"""Exact unique name_core rule for queries the pipeline leaves unassigned (validation)."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W = "../work/"
T = {"US": .75, "India": .8}
s1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country","name_core","state","addr_empty"]).rename({"entity_id":"s1_id","name_core":"n","state":"st1","addr_empty":"e1"})
cnt = s1.group_by("country","n").len("ncnt")
s1 = s1.join(cnt, on=["country","n"])
q = pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet", columns=["entity_id","country","name_core","state","addr_empty"]) for k in (2,3)]).rename({"entity_id":"s23_id","name_core":"n","state":"stq","addr_empty":"eq"})
gt = pl.read_parquet(W+"gt_pairs.parquet").with_columns(pl.lit(1).alias("lab"))
v = pl.read_parquet(W+"valid_scores_stage2.parquet").join(s1.select("s1_id","country"), on="s1_id")
top = v.sort("p2", descending=True).unique("s23_id", keep="first")
pred = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T)).select("s23_id", pl.col("s1_id").alias("pred_s1"))
# queries whose exact name hits a unique valid S1
m = q.join(s1.filter(is_valid_expr("s1_id")), on=["country","n"]).filter(pl.col("ncnt")==1)
m = m.join(gt, on=["s1_id","s23_id"], how="left").with_columns(pl.col("lab").fill_null(0))
m = m.join(pred, on="s23_id", how="left")
# how many S2/S3 queries share this exact name in country (crowding)
qc = q.group_by("country","n").len("qcnt"); m = m.join(qc, on=["country","n"])
m = m.with_columns(pl.when(pl.col("pred_s1").is_null()).then(pl.lit("unassigned")).when(pl.col("pred_s1")==pl.col("s1_id")).then(pl.lit("same")).otherwise(pl.lit("other")).alias("st"),
    (pl.col("stq").is_null() | pl.col("st1").is_null() | (pl.col("stq")==pl.col("st1"))).alias("state_ok"))
print(m.group_by("country","st","eq").agg(pl.len(), pl.col("lab").mean().round(4).alias("prec")).sort("country","st","eq"))
u = m.filter(pl.col("st")=="unassigned")
print(u.group_by("country","eq","state_ok",pl.col("qcnt").clip(upper_bound=6)).agg(pl.len(), pl.col("lab").sum().alias("tp"), pl.col("lab").mean().round(3).alias("prec")).sort("country","eq","state_ok","qcnt"))
u.write_parquet(W+"v10/exact_unassigned.parquet")
