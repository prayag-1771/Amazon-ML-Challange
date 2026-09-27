import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
pl.Config.set_tbl_rows(60)
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
pred=pl.read_parquet("pred15_v.parquet"); c=pl.read_parquet("trans_cp.parquet").filter("s1eq")
m0=macro_f05(s1["s1_id"],pred,gt,by=s1)
print(c.with_columns(pl.col("vp2").cut([.01,.1,.3,.5]).alias("b")).group_by("topagree","b").agg(pl.len(),pl.col("y").mean().round(3)).sort("topagree","b"))
print(c.group_by("s1e","src").agg(pl.len(),pl.col("y").mean().round(3)))
print(c.with_columns(pl.col("s1_ncnt").clip(0,8)).group_by("s1_ncnt","nacc").agg(pl.len(),pl.col("y").mean().round(3)).filter(pl.col("len")>20).sort("s1_ncnt","nacc"))
def ev(name,a):
    m=macro_f05(s1["s1_id"],pl.concat([pred,a.select("s1_id","s23_id",pl.lit("tr").alias("ch"))]),gt,by=s1)
    print(name,a.height,round(a["y"].mean(),3),round(m["f05"]-m0["f05"],5),round(m["f05_US"]-m0["f05_US"],5),round(m["f05_India"]-m0["f05_India"],5))
for K in (4,8,12,20,1000):
    ev(f"ncnt<={K}",c.filter(pl.col("s1_ncnt")<=K))
    ev(f"ncnt<={K}|topagree",c.filter((pl.col("s1_ncnt")<=K)|pl.col("topagree")))
