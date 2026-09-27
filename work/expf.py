import sys; sys.path.insert(0,'../business_entity_resolution')
import numpy as np, polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
P=pl.read_parquet("pred15_v.parquet"); pred=P.select("s1_id","s23_id")
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
v=pl.read_parquet("valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first").filter(pl.col("p2")>=0.02)
# calibration

top=top.with_columns(pl.col("p2").cast(pl.Float64)).join(pred.select("s23_id"),on="s23_id",how="anti")  # sub-threshold candidates not already predicted
cur=pred.join(v.select("s1_id","s23_id","p2"),on=["s1_id","s23_id"],how="left").with_columns(pl.col("p2").cast(pl.Float64).fill_null(0.97),pl.lit(1).alias("fix"))
allc=pl.concat([cur.select("s1_id","s23_id","p2","fix"),top.select("s1_id","s23_id","p2",pl.lit(0).alias("fix"))])
m0=macro_f05(s1["s1_id"],pred,gt,by=s1); print("base",round(m0["f05"],5),round(m0["f05_US"],5),round(m0["f05_India"],5))
def run(scale):
    adds=[]
    for g in allc.filter(pl.col("fix")==0).select("s1_id").unique().join(allc,on="s1_id").sort(["s1_id","fix","p2"],descending=[False,True,True]).partition_by("s1_id"):
        p=g["p2"].to_numpy()*scale; fix=g["fix"].to_numpy(); nf=int(fix.sum()); nt=p.sum()
        best,bk=-1,nf
        for k in range(nf,len(p)+1):
            if k==0: val=np.prod(1-p)
            else: tp=p[:k].sum(); val=1.25*tp/(0.25*nt+k)
            if val>best: best,bk=val,k
        if bk>nf: adds.append(g[nf:bk].select("s1_id","s23_id"))
    a=pl.concat(adds) if adds else pl.DataFrame(schema={"s1_id":pl.Utf8,"s23_id":pl.Utf8})
    lab=a.join(gt.with_columns(pl.lit(1).alias("l")),on=["s1_id","s23_id"],how="left")["l"].fill_null(0).mean()
    m=macro_f05(s1["s1_id"],pl.concat([pred,a]),gt,by=s1)
    print(scale,a.height,lab,round(m["f05"],5),round(m["f05_US"],5),round(m["f05_India"],5))
for sc in (1.0,0.8,0.6): run(sc)
