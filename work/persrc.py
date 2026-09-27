import polars as pl
W="../work/"
gt = pl.read_parquet(W+"gt_pairs.parquet")
src = pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet", columns=["entity_id"]).with_columns(pl.lit(k).alias("src")) for k in (2,3)]).rename({"entity_id":"s23_id"})
g = gt.join(src, on="s23_id").group_by("s1_id").agg((pl.col("src")==2).sum().alias("n2"), (pl.col("src")==3).sum().alias("n3"))
print(g.group_by("n2","n3").len().sort("len", descending=True).head(25))
