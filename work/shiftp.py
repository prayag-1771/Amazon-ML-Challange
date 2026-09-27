import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
pl.Config.set_tbl_rows(40)
gt=load_ground_truth()
s1v=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
qtr=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','country','addr_empty']) for k in (2,3)]).rename({'entity_id':'s23_id'})
# validation query universe: queries whose truth S1 is valid, or singleton queries assigned... approximate: queries in val scores + queries with valid truth
v=pl.read_parquet("valid_scores_stage2.parquet")
vtop=v.sort("p2",descending=True).unique("s23_id",keep="first").join(s1v,on="s1_id")
t=pl.read_parquet("top15_test.parquet")
bins=[.02,.05,.1,.2,.3,.5,.6,.7,.75,.8,.9,.97]
def h(d,n,name):
    return d.with_columns(pl.col("p2").cut(bins).alias("bin")).group_by("country","bin").len().join(n,on="country").with_columns((pl.col("len")/pl.col("n")*1e4).round(1).alias(name)).select("country","bin",name)
nv=s1v.group_by("country").len("n"); nt=pl.read_parquet("norm_test_s1.parquet",columns=["country"]).group_by("country").len("n")
x=h(vtop,nv,"val_per1e4S1").join(h(t,nt,"test_per1e4S1"),on=["country","bin"],how="full",coalesce=True).filter(pl.col("country")!="France").sort("country","bin")
x=x.with_columns((pl.col("test_per1e4S1")/pl.col("val_per1e4S1")).round(2).alias("ratio"))
print(x)
