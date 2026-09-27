import sys; sys.path.insert(0, ".")
import polars as pl
from src.blocking import _frames
W = "../work/"
g = pl.read_parquet(W + "fnb.parquet").filter(pl.col("cls").is_in(["no_cands", "not_in_cands"]))
s1, q = _frames("train")
g = g.join(s1.select(pl.col("s1_row").cast(pl.UInt32), pl.col("entity_id").alias("s1_id")), on="s1_id").join(
    q.select(pl.col("q_row").cast(pl.UInt32), pl.col("entity_id").alias("s23_id")), on="s23_id")
c = pl.read_parquet(W + "cand_train.parquet", columns=["q_row", "s1_row", "emb_rank", "tf_rank"])
p = pl.read_parquet(W + "pruned_train.parquet", columns=["q_row", "s1_row"])
g = g.join(c.with_columns(pl.lit(1).alias("in_union")), on=["q_row", "s1_row"], how="left").join(
    p.with_columns(pl.lit(1).alias("in_pruned")), on=["q_row", "s1_row"], how="left")
g = g.join(p.group_by("q_row").len("npr"), on="q_row", how="left")
print(g.group_by("country", "q_empty", pl.col("in_union").is_not_null().alias("union"), pl.col("in_pruned").is_not_null().alias("pruned"), pl.col("npr").is_not_null().alias("q_has_pruned")).len().sort("country", "q_empty", "union").rows())
g.write_parquet(W + "fnb2.parquet")
