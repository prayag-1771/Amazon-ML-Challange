import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth().join(s1v, on="s1_id")
d = pl.read_parquet(W + "valid_scores_stage2.parquet")
cand = pl.read_parquet(W + "valid_feats.parquet", columns=["s1_id", "s23_id", "q_addr_empty"])
q = pl.concat([pl.read_parquet(W + f"norm_train_s{i}.parquet", columns=["entity_id", "addr_core", "name_core"]) for i in (2, 3)]).rename({"entity_id": "s23_id"})
top = d.sort("p2", descending=True).unique("s23_id", keep="first").join(s1v.select("s1_id","country"), on="s1_id")
T = {"US": 0.75, "India": 0.80}
acc = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T)).select("s1_id", "s23_id", pl.lit(1).alias("acc"))
g = gt.join(d.select("s1_id", "s23_id", "p2"), on=["s1_id", "s23_id"], how="left").join(acc, on=["s1_id", "s23_id"], how="left")
g = g.join(top.select("s23_id", pl.col("s1_id").alias("top_s1"), pl.col("p2").alias("top_p2")), on="s23_id", how="left").join(q, on="s23_id", how="left")
g = g.with_columns(pl.when(pl.col("acc") == 1).then(pl.lit("TP")).when(pl.col("p2").is_null() & pl.col("top_s1").is_null()).then(pl.lit("no_cands"))
    .when(pl.col("p2").is_null()).then(pl.lit("not_in_cands")).when(pl.col("top_s1") != pl.col("s1_id")).then(pl.lit("lost_to_other")).otherwise(pl.lit("low_score")).alias("cls"),
    (pl.col("addr_core").fill_null("").str.strip_chars() == "").alias("q_empty"))
print(g.group_by("country", "cls").len().sort("country", "len"))
print(g.filter(pl.col("cls") != "TP").group_by("country", "cls", "q_empty").len().sort("country", "cls", "q_empty"))
print(g.group_by("country").agg(pl.col("q_empty").mean()))
g.write_parquet(W + "fnb.parquet")
