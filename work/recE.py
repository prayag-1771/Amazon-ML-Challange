import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1,on="s1_id")
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','addr_empty']) for k in (2,3)]).rename({'entity_id':'s23_id'})
pred=pl.read_parquet("pred15_v.parquet")
g=gt.join(q,on="s23_id").with_columns(pl.struct("s1_id","s23_id").is_in(pred.select(pl.struct("s1_id","s23_id")).to_series().implode()).alias("hit"))
print(g.group_by("country","addr_empty").agg(pl.len(),pl.col("hit").mean().round(4)).sort("country","addr_empty"))
