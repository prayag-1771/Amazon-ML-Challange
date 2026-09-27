import sys; sys.path.insert(0,'.')
import polars as pl
from src import india_addr as ia
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
gt=load_ground_truth()
hit=gt.join(pl.read_parquet(ia.W+"pruned_train.parquet",columns=["s1_id","s23_id"]),on=["s1_id","s23_id"],how="semi")
d=ia.pairs("train",hit).join(gt.with_columns(pl.lit(1,pl.Int8).alias("label")),on=["s1_id","s23_id"],how="left")
d=d.with_columns(pl.col("label").fill_null(0),is_valid_expr("s1_id").alias("vs1"))
vq=d.filter(pl.col("vs1")).select("q_row").unique()
va=ia.score(d.join(vq,on="q_row",how="semi"))
va.write_parquet(ia.W+"india_addr_valid_scored.parquet")
print(va.shape)
