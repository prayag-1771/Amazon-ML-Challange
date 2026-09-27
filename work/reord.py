import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
def load(split):
    s1=pl.read_parquet(f'norm_{split}_s1.parquet',columns=['entity_id','country','name_core','addr_core']).rename({'entity_id':'s1_id','name_core':'n1','addr_core':'a1'})
    q=pl.concat([pl.read_parquet(f'norm_{split}_s{k}.parquet',columns=['entity_id','name_core','addr_core']) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','addr_core':'aq'})
    return s1,q
srt=lambda c: pl.col(c).fill_null("").str.split(" ").list.sort().list.join(" ")
def cls(t):
    return t.with_columns(pl.when(pl.col("a1")==pl.col("aq")).then(pl.lit("same")).when(srt("a1")==srt("aq")).then(pl.lit("reord")).otherwise(pl.lit("diff")).alias("ak"),
                          (pl.col("n1")==pl.col("nq")).alias("neq"))
s1,q=load('test')
t=pl.read_parquet('test_scores_stage2_v15.parquet').sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id')
t=cls(t).filter(pl.col("aq")!="")
print(t.group_by("country","ak").agg(pl.len(),(pl.col("p2")>=.75).mean().round(3).alias("hi"),pl.col("p2").is_between(.1,.75).mean().round(3).alias("mid")).sort("country","ak"))
print(t.filter((pl.col("ak")=="reord")).group_by("country","neq").agg(pl.len(),(pl.col("p2")>=.75).mean().round(3).alias("hi"),pl.col("p2").is_between(.1,.75).mean().round(3).alias("mid")).sort("country","neq"))
s1,q=load('train'); s1=s1.filter(is_valid_expr('s1_id'))
v=pl.read_parquet('valid_scores_stage2.parquet').sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id')
v=cls(v).filter(pl.col("aq")!="")
print(v.group_by("country","ak").agg(pl.len(),pl.col("label").mean().round(3),(pl.col("p2")>=.75).mean().round(3).alias("hi"),pl.col("p2").is_between(.1,.75).mean().round(3).alias("mid")).sort("country","ak"))
