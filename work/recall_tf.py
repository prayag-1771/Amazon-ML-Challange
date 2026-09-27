import polars as pl
from src.blocking import with_ids
from src.io_utils import load_ground_truth
from src.normalize import normalize_source
tf = pl.read_parquet("../work/cand_tf_train.parquet").with_columns(pl.col("q_row","s1_row").cast(pl.UInt32))
print("pairs", len(tf), "queries", tf["q_row"].n_unique())
tf = with_ids(tf, "train")
gt = load_ground_truth()
c = normalize_source("train",1).select(pl.col("entity_id").alias("s1_id"),"country")
h = gt.join(c,on="s1_id").join(tf.select("s1_id","s23_id","tf_rank"),on=["s1_id","s23_id"],how="left")
print(h.group_by("country").agg(pl.len(), *[(pl.col("tf_rank")<k).fill_null(False).mean().alias(f"@{k}") for k in (1,3,5)]))
