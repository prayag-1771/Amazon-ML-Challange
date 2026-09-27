import sys; sys.path.insert(0, ".")
import polars as pl
W="../work/"
miss=pl.read_parquet(W+"miss_v12.parquet")
s1=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id"]).with_row_index("s1_row").rename({"entity_id":"s1_id"})
q=pl.concat([pl.read_parquet(W+f"norm_train_s{i}.parquet",columns=["entity_id"]) for i in (2,3)]).with_row_index("q_row").rename({"entity_id":"s23_id"})
m=miss.join(s1,on="s1_id").join(q,on="s23_id").with_columns(pl.col("s1_row","q_row").cast(pl.UInt32))
for f in ("cand_tf_train","cand_emb_train","cand_train","pruned_train"):
    c=pl.read_parquet(W+f+".parquet",columns=["q_row","s1_row"]).with_columns(pl.col("q_row","s1_row").cast(pl.UInt32))
    m=m.join(c.with_columns(pl.lit(True).alias(f)),on=["q_row","s1_row"],how="left").with_columns(pl.col(f).fill_null(False))
print(m.group_by("country","qe","cand_train","pruned_train").len().sort("country","qe","cand_train").rows())
