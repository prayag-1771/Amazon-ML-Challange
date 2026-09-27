import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
P=pl.read_parquet("pred15_v.parquet")
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
v=pl.read_parquet("valid_scores_stage2.parquet").sort("p2",descending=True).unique("s23_id",keep="first").join(s1,on="s1_id")
ch=P.filter(pl.col("ch")!="base").select("s1_id","s23_id")
base=P.filter(pl.col("ch")=="base").select("s1_id","s23_id")
# base reproduced = v top with p2>=T?
for tu in (.65,.7,.75,.8):
  for ti in (.7,.75,.8,.85):
    b=v.filter(pl.col("p2")>=pl.when(pl.col("country")=="US").then(tu).otherwise(ti)).select("s1_id","s23_id")
    m=macro_f05(s1["s1_id"],pl.concat([b,ch]),gt,by=s1)
    print(tu,ti,round(m["f05"],5),round(m["f05_US"],5),round(m["f05_India"],5))
