import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W="../work/"; T={"US":.75,"India":.8}
s1=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
v=pl.read_parquet(W+"valid_scores_stage2.parquet").join(s1,on="s1_id")
vf=pl.read_parquet(W+"valid_feats.parquet",columns=["s1_id","s23_id","q_addr_empty","s1_name_cnt","nc_eq","num_first_eq","state_eq","nc_ratio","ac_ratio"])
top=v.sort("p2",descending=True).unique("s23_id",keep="first").join(vf,on=["s1_id","s23_id"],how="left")
th=pl.col("country").replace_strict(T)
fn=top.filter((pl.col("label")==1)&(pl.col("p2")<th)); fp=top.filter((pl.col("label")==0)&(pl.col("p2")>=th))
nb=top.filter((pl.col("label")==0)&(pl.col("p2")<th)&(pl.col("p2")>=0.3))
print("FN below thr", fn.group_by("country").len().rows(), "FP", fp.group_by("country").len().rows())
for nm,x in (("FN",fn),("negatives p2 .3-thr",nb)):
    print(nm)
    print(x.group_by("country","q_addr_empty").agg(pl.len(),pl.col("p2").mean().round(2),(pl.col("s1_name_cnt")>1).mean().round(2).alias("dupname"),pl.col("nc_eq").mean().round(2),pl.col("num_first_eq").mean().round(2),pl.col("nc_ratio").mean().round(1),pl.col("ac_ratio").mean().round(1)).sort("country","q_addr_empty").rows())
print("FN p2 dist", fn.with_columns(pl.col("p2").cut([.05,.2,.4,.6]).alias("b")).group_by("country","b").len().sort("country","b").rows())
