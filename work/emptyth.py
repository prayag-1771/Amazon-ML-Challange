import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W="../work/"
s1=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
v=pl.read_parquet(W+"valid_scores_stage2.parquet").join(s1,on="s1_id")
qe=pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet",columns=["entity_id","addr_empty"]) for k in (2,3)]).rename({"entity_id":"s23_id"})
top=v.sort("p2",descending=True).unique("s23_id",keep="first").join(qe,on="s23_id")
T={"US":.75,"India":.8}
th=pl.col("country").replace_strict(T)
b=top.filter(pl.col("p2")>=th); m0=macro_f05(s1["s1_id"],b.select("s1_id","s23_id"),gt,by=s1)
print("base",round(m0["f05"],5),round(m0["f05_US"],5),round(m0["f05_India"],5))
e=top.filter(pl.col("addr_empty")==1)
print(e.with_columns(pl.col("p2").cut([.2,.4,.5,.6,.7,.75,.8,.9,.97]).alias("b")).group_by("country","b").agg(pl.len(),pl.col("label").mean().round(3)).sort("country","b").rows())
for te in (.4,.5,.6,.65,.7,.85,.9):
    x=top.filter(pl.when(pl.col("addr_empty")==1).then(pl.col("p2")>=te).otherwise(pl.col("p2")>=th))
    m=macro_f05(s1["s1_id"],x.select("s1_id","s23_id"),gt,by=s1)
    print(te,{k:round(m[k]-m0[k],5) for k in ("f05","f05_US","f05_India")})
