import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
pl.Config.set_tbl_rows(40)
T={"US":.75,"India":.8}
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','addr_empty']) for k in (2,3)]).rename({'entity_id':'s23_id'})
v=pl.read_parquet("valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first")
base=top.filter(pl.col("p2")>=pl.col("country").replace_strict(T)).select("s1_id","s23_id").with_columns(pl.lit("base").alias("ch"))
r=pl.read_parquet("v10/rescue_valid.parquet").filter(pl.col("vs1")).sort("pr",descending=True).unique("s23_id",keep="first").filter(pl.col("pr")>=.9).join(v.select("s23_id").unique(),on="s23_id",how="anti").select("s1_id","s23_id").with_columns(pl.lit("rescue").alias("ch"))
casc=pl.read_parquet("cascade_valid_p001.parquet").select("s23_id").unique()
b=pl.concat([base,r])
a=pl.read_parquet("v10/ac2_valid_mid.parquet").sort("pr",descending=True).unique("q_row",keep="first").filter(pl.col("vs1")&(pl.col('pr')>=.9)).join(casc,on="s23_id",how="anti").join(b,on="s23_id",how="anti").select("s1_id","s23_id").with_columns(pl.lit("inaddr").alias("ch"))
pred=pl.concat([b,a])
pred.write_parquet("pred15_v.parquet")
m=macro_f05(s1["s1_id"],pred,gt,by=s1); print(m)
# per-S1 loss attribution
tp=pred.join(gt,on=["s1_id","s23_id"]).group_by("s1_id").len("tp")
npd=pred.group_by("s1_id").len("np"); nt=gt.group_by("s1_id").len("nt")
d=s1.join(tp,on="s1_id",how="left").join(npd,on="s1_id",how="left").join(nt,on="s1_id",how="left").fill_null(0)
d=d.with_columns(pl.when(pl.col("nt")==0).then((pl.col("np")==0).cast(pl.Float64)).when(pl.col("tp")==0).then(0.0).otherwise(1.25*pl.col("tp")/(pl.col("np")+.25*pl.col("nt"))).alias("f"))
d=d.with_columns(pl.when(pl.col("nt")==0).then(pl.lit("singletonFP")).when(pl.col("tp")==0).then(pl.when(pl.col("np")==0).then(pl.lit("allmiss")).otherwise(pl.lit("allwrong"))).when((pl.col("np")>pl.col("tp"))&(pl.col("nt")>pl.col("tp"))).then(pl.lit("fp+fn")).when(pl.col("np")>pl.col("tp")).then(pl.lit("fp_only")).when(pl.col("nt")>pl.col("tp")).then(pl.lit("fn_only")).otherwise(pl.lit("ok")).alias("kind"))
N=d.height
print(d.group_by("country","kind").agg(pl.len(),((1-pl.col("f")).sum()/N*1e4).round(2).alias("loss_bp")).sort("loss_bp",descending=True))
d.write_parquet("loss15_v.parquet")
