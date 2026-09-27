import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
T={"US":.75,"India":.8}
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
v=pl.read_parquet("valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first").with_columns((pl.col("p2")>=pl.col("country").replace_strict(T)).alias("acc"))
base=top.filter("acc").select("s1_id","s23_id")
nacc=base.group_by("s1_id").len("nacc")
cand=top.filter(~pl.col("acc")).join(nacc,on="s1_id",how="left").fill_null(0)
m0=macro_f05(s1["s1_id"],base,gt,by=s1); print("base",round(m0['f05'],5),round(m0['f05_US'],5),round(m0['f05_India'],5))
for ks in ([1],[1,2],[1,2,3,4,5,6,7,8,9,99]):
  for lo in (.5,.6,.65,.7):
    add=cand.filter(pl.col("nacc").is_in(ks)&(pl.col("p2")>=lo))
    # only one extra per S1: best
    add1=add.sort("p2",descending=True).unique("s1_id",keep="first")
    for nm,a in (("all",add),("one",add1)):
      m=macro_f05(s1["s1_id"],pl.concat([base,a.select("s1_id","s23_id")]),gt,by=s1)
      print(ks[-1],lo,nm,a.height,round(a['label'].mean(),3),round(m['f05']-m0['f05'],5),round(m['f05_US']-m0['f05_US'],5),round(m['f05_India']-m0['f05_India'],5))
