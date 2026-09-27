import polars as pl
from src.io_utils import load_source
sc=pl.read_parquet("../work/test_scores_lgb_v6k.parquet")
s1=load_source("test",1).select(pl.col("entity_id").alias("s1_id"),pl.col("business_name").alias("n1"),pl.col("business_address").alias("a1"),"country")
q=pl.concat([load_source("test",s) for s in (2,3)]).select(pl.col("entity_id").alias("s23_id"),pl.col("business_name").alias("nq"),pl.col("business_address").alias("aq"),"country")
top=sc.sort("p",descending=True).unique("s23_id",keep="first")
allq=q.join(top,on="s23_id",how="left")
print("no candidates at all:", allq.group_by("country").agg(pl.col("p").is_null().mean(), pl.len()))
x=allq.filter(pl.col("country")=="US").join(s1.drop("country"),on="s1_id",how="left")
low=x.filter(pl.col("p")<0.02).sample(25,seed=3)
for r in low.iter_rows(named=True):
    print(f"p={r['p']:.3f}\n  S1: {r['n1']} | {r['a1']}\n  Q : {r['nq']} | {r['aq']}")
nc=x.filter(pl.col("p").is_null()).sample(10,seed=3)
for r in nc.iter_rows(named=True): print("NOCAND Q:", r['nq'],"|",r['aq'])
