import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
T={"US":.75,"India":.8}
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','addr_empty']) for k in (2,3)]).rename({'entity_id':'s23_id'})
v=pl.read_parquet("valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first").join(q,on='s23_id')
r=pl.read_parquet("v10/rescue_valid.parquet").filter(pl.col("vs1")).sort("pr",descending=True).unique("s23_id",keep="first").filter(pl.col("pr")>=.9).join(v.select("s23_id").unique(),on="s23_id",how="anti").select("s1_id","s23_id")
casc=pl.read_parquet("cascade_valid_p001.parquet").select("s23_id").unique()
def run(tE):
    th=pl.when(pl.col('addr_empty')==1).then(pl.col('country').replace_strict(tE)).otherwise(pl.col("country").replace_strict(T))
    base=top.filter(pl.col("p2")>=th).select("s1_id","s23_id")
    b=pl.concat([base,r])
    a=pl.read_parquet("v10/ac2_valid_mid.parquet").sort("pr",descending=True).unique("q_row",keep="first").filter(pl.col("vs1")&(pl.col('pr')>=.9)).join(casc,on="s23_id",how="anti").join(b,on="s23_id",how="anti")
    return macro_f05(s1["s1_id"],pl.concat([b,a.select("s1_id","s23_id")]),gt,by=s1)
for us in (.5,.6,.75,.85):
    for ind in (.5,.65,.8,.9):
        m=run({'US':us,'India':ind}); print(us,ind,{k:round(x,5) for k,x in m.items() if k in('f05','f05_US','f05_India')})
