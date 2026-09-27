import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
pred=pl.read_parquet("pred15_v.parquet").select("s1_id","s23_id")
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
v=pl.read_parquet("valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first").join(pred,on="s23_id",how="anti")
lon=top.join(pred,on="s1_id",how="anti")  # S1 has no predicted match at all
m0=macro_f05(s1["s1_id"],pred,gt,by=s1); print("base",round(m0["f05"],5),round(m0["f05_US"],5),round(m0["f05_India"],5))
for one in (True,False):
  for t in (.3,.4,.5,.6,.7):
    a=lon.filter(pl.col("p2")>=t)
    if one: a=a.sort("p2",descending=True).unique("s1_id",keep="first")
    m=macro_f05(s1["s1_id"],pl.concat([pred,a.select("s1_id","s23_id")]),gt,by=s1)
    print(one,t,a.height,round(a["label"].mean(),3),round(m["f05"],5),round(m["f05_US"],5),round(m["f05_India"],5))
