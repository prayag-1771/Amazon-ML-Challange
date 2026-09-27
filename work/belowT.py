import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
pl.Config.set_tbl_rows(60)
T={"US":.75,"India":.8}
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country","addr_empty"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id","addr_empty":"s1e"})
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','addr_empty']) for k in (2,3)]).rename({'entity_id':'s23_id'})
v=pl.read_parquet("valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first").join(q,on="s23_id")
top=top.with_columns(pl.col("p2").cut([.3,.5,.75]).alias("b"))
print(top.filter(pl.col("p2")>=.3).group_by("country","addr_empty","s1e","b").agg(pl.len(),pl.col("label").sum().alias("pos"),pl.col("label").mean().round(3)).sort("country","addr_empty","s1e","b"))
