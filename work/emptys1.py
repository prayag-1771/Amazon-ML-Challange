"""Macro-F-aware decoding: an S1 with no accepted match scores 0 unless it is a true singleton. For such S1s,
accept the best unassigned query whose top-1 is this S1 at a lower threshold."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W="../work/"; T={"US":.75,"India":.8}
s1=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
v=pl.read_parquet(W+"valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first")
th=pl.col("country").replace_strict(T)
acc=top.filter(pl.col("p2")>=th)
r=pl.read_parquet(W+"v10/rescue_valid.parquet").filter(pl.col("vs1")).sort("pr",descending=True).unique("s23_id",keep="first").filter(pl.col("pr")>=.9).join(v.select("s23_id").unique(),on="s23_id",how="anti")
base=pl.concat([acc.select("s1_id","s23_id"),r.select("s1_id","s23_id")])
m0=macro_f05(s1["s1_id"],base,gt,by=s1); print("v13",{k:round(x,5) for k,x in m0.items()})
has=base.select("s1_id").unique()
rej=top.filter(pl.col("p2")<th).join(has,on="s1_id",how="anti")
b1=rej.sort("p2",descending=True).unique("s1_id",keep="first")
print(b1.with_columns(pl.col("p2").cut([.1,.2,.3,.4,.5,.6,.7]).alias("b")).group_by("country","b").agg(pl.len(),pl.col("label").mean().round(3)).sort("country","b").rows())
for t0 in (.2,.3,.4,.5,.6):
    add=b1.filter(pl.col("p2")>=t0)
    m=macro_f05(s1["s1_id"],pl.concat([base,add.select("s1_id","s23_id")]),gt,by=s1)
    print(t0,len(add),round(add["label"].mean(),3),{k:round(m[k]-m0[k],5) for k in ("f05","f05_US","f05_India")})
# also: second-rank S1 of assigned queries? skip. Also S1 with exactly one match: accept 2nd?
