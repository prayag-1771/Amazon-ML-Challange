import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W="../work/"
nm = pl.read_parquet(W+"namechan_train_empty.parquet")
s1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id"]).with_row_index("s1_row")
q = pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet", columns=["entity_id","country","addr_empty"]) for k in (2,3)]).with_row_index("q_row").filter(pl.col("addr_empty")==1)
gt = pl.read_parquet(W+"gt_pairs.parquet").filter(is_valid_expr("s1_id"))
g = gt.join(q.select(pl.col("q_row").cast(pl.UInt32), pl.col("entity_id").alias("s23_id"), "country"), on="s23_id").join(s1.select(pl.col("s1_row").cast(pl.UInt32), pl.col("entity_id").alias("s1_id")), on="s1_id")
raw = pl.read_parquet(W+"cand_train.parquet", columns=["q_row","s1_row"]).with_columns(pl.lit(True).alias("raw"))
pr = pl.read_parquet(W+"pruned_train.parquet", columns=["q_row","s1_row"]).with_columns(pl.lit(True).alias("pr"))
g = g.join(raw, on=["q_row","s1_row"], how="left").join(pr, on=["q_row","s1_row"], how="left").join(nm, on=["q_row","s1_row"], how="left")
g = g.with_columns(pl.col("raw","pr").fill_null(False), pl.col("rk").fill_null(999))
for k in (1, 3, 5, 10, 20):
    print(k, g.group_by("country").agg(pl.len(), pl.col("pr").mean().alias("pruned"), (pl.col("rk")<=k).mean().alias("name_k"),
          ((pl.col("rk")<=k)|pl.col("pr")).mean().alias("pr_or_name")).sort("country").rows())
print("cands per empty query at k=10:", nm.filter(pl.col("rk")<=10).group_by("q_row").len()["len"].mean())
