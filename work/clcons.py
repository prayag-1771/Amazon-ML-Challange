import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
pred=pl.read_parquet("pred15_v.parquet").select("s1_id","s23_id")
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
q=pl.concat([pl.read_parquet(f"norm_train_s{k}.parquet",columns=["entity_id","name_core","addr_core"]) for k in (2,3)]).rename({"entity_id":"s23_id"}).with_columns(pl.col("name_core").fill_null(""),pl.col("addr_core").fill_null(""))
v=pl.read_parquet("valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first").join(pred,on="s23_id",how="anti").join(q,on="s23_id")
pm=pred.join(q,on="s23_id").select("s1_id",pl.col("name_core").alias("n2"),pl.col("addr_core").alias("a2")).unique()
m0=macro_f05(s1["s1_id"],pred,gt,by=s1); print("base",round(m0["f05"],5))
for key,cond in (("name",pl.col("name_core")==pl.col("n2")),("name+addr",(pl.col("name_core")==pl.col("n2"))&(pl.col("addr_core")==pl.col("a2"))),("addr",(pl.col("addr_core")==pl.col("a2"))&(pl.col("addr_core")!=""))):
  x=top.join(pm,on="s1_id").filter(cond).select("s1_id","s23_id","p2","label","country").unique()
  for t in (.05,.2,.4):
    a=x.filter(pl.col("p2")>=t)
    m=macro_f05(s1["s1_id"],pl.concat([pred,a.select("s1_id","s23_id")]),gt,by=s1)
    print(key,t,a.height,round(a["label"].mean(),3),round(m["f05"],5),round(m["f05_US"],5),round(m["f05_India"],5))
