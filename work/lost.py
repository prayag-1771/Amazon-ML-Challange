import sys
sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.blocking import with_ids
W = "../work/"
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = pl.read_parquet(W + "gt_pairs.parquet").join(s1, on="s1_id")
pr = pl.read_parquet(W + "pruned_train.parquet", columns=["s1_id", "s23_id", "p0"])
full = with_ids(pl.read_parquet(W + "cand_train.parquet"), "train").select("s1_id", "s23_id", "emb_rank", "tf_rank")
full = full.join(gt.select("s23_id"), on="s23_id", how="semi")
q = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=["entity_id", "addr_empty"]) for k in (2, 3)]).rename({"entity_id": "s23_id"})
d = gt.join(pr.select("s1_id", "s23_id", pl.lit(1).alias("in_pr")), on=["s1_id", "s23_id"], how="left") \
      .join(full.with_columns(pl.lit(1).alias("in_blk")), on=["s1_id", "s23_id"], how="left").join(q, on="s23_id")
d = d.with_columns(pl.when(pl.col("in_pr") == 1).then(pl.lit("kept")).when(pl.col("in_blk") == 1).then(pl.lit("lost_prune")).otherwise(pl.lit("lost_block")).alias("k"))
print(d.group_by("country", "addr_empty", "k").len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("share_of_country")).sort("country", "addr_empty", "k"))
