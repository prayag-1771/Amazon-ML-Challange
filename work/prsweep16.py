import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
pred=pl.read_parquet("pred15_v.parquet"); b=pred.filter(pl.col("ch")!="inaddr").select("s1_id","s23_id")
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
casc=pl.read_parquet("cascade_valid_p001.parquet").select("s23_id").unique()
A=pl.read_parquet("ia16_valid.parquet").sort("pr",descending=True).unique("q_row",keep="first").filter(pl.col("vs1")).join(casc,on="s23_id",how="anti").join(b,on="s23_id",how="anti")
for th in (.5,.6,.7,.8,.85,.9,.95):
    a=A.filter(pl.col("pr")>=th)
    m=macro_f05(s1["s1_id"],pl.concat([b,a.select("s1_id","s23_id")]),gt,by=s1)
    print(th,a.height,round(a["label"].mean(),3),round(m["f05"],5),round(m["f05_India"],5))
