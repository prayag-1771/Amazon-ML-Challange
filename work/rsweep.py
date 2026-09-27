import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
pred=pl.read_parquet("pred15_v.parquet"); b=pred.filter(pl.col("ch")!="rescue").select("s1_id","s23_id")
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
v=pl.read_parquet("valid_scores_stage2.parquet",columns=["s23_id"]).unique()
R=pl.read_parquet("v10/rescue_valid.parquet").filter(pl.col("vs1")).sort("pr",descending=True).unique("s23_id",keep="first").join(v,on="s23_id",how="anti").join(b,on="s23_id",how="anti").join(s1,on="s1_id")
for c in ("US","India"):
  for th in (.6,.7,.8,.85,.9,.95):
    r=R.filter((pl.col("pr")>=th)&((pl.col("country")==c)|(pl.col("pr")>=.9)))
    m=macro_f05(s1["s1_id"],pl.concat([b,r.select("s1_id","s23_id")]),gt,by=s1)
    print(c,th,r.height,round(m["f05"],5),round(m["f05_US"],5),round(m["f05_India"],5))
