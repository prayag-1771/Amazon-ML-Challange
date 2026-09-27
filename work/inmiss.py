import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_width_chars(260)
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country","name_core","addr_core","state"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id","name_core":"n1","addr_core":"a1","state":"st1"})
gt=load_ground_truth().join(s1,on="s1_id")
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','addr_empty','name_core','addr_core','state']) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','addr_core':'aq','state':'stq'})
pred=pl.read_parquet("pred15_v.parquet")
v=pl.read_parquet("valid_scores_stage2.parquet")
m=gt.join(q,on="s23_id").filter(pl.col("addr_empty")==0).join(pred,on=["s1_id","s23_id"],how="anti")
m=m.join(v.select("s1_id","s23_id","p2"),on=["s1_id","s23_id"],how="left")
m=m.with_columns(pl.when(pl.col("p2").is_null()).then(pl.lit("nocand")).when(pl.col("p2")<.1).then(pl.lit("low")).otherwise(pl.lit("mid")).alias("k"),(pl.col("st1")==pl.col("stq")).alias("steq"))
print(m.group_by("country","k").agg(pl.len(),pl.col("steq").mean().round(3)).sort("country","k"))
print(m.filter((pl.col("country")=="India")&(pl.col("k")=="nocand")).sample(25,seed=4).select("nq","n1","aq","a1"))
print(m.filter((pl.col("country")=="US")&(pl.col("k")!="nocand")).sample(20,seed=4).select("nq","n1","aq","a1","p2"))
sys.path.insert(0,'../business_entity_resolution'); from src.india_addr import skel
mm=m.with_columns((pl.col("nq").str.replace_all(" ","")==pl.col("n1").str.replace_all(" ","")).alias("ceq"),
    skel("nq").alias("kq"),skel("n1").alias("k1"))
mm=mm.with_columns((pl.col("kq").str.replace_all(" ","")==pl.col("k1").str.replace_all(" ","")).alias("keq"),(pl.col("nq")==pl.col("n1")).alias("neq"))
print(mm.group_by("country","k").agg(pl.len(),pl.col("neq").sum(),pl.col("ceq").sum(),pl.col("keq").sum()).sort("country","k"))
a=pl.read_parquet("v10/ac2_valid_mid.parquet"); print(a.columns)
x=mm.filter((pl.col("country")=="India")&(pl.col("k")=="nocand")).join(a.select("s1_id","s23_id","pr"),on=["s1_id","s23_id"],how="left")
top=a.sort("pr",descending=True).unique("s23_id",keep="first").select("s23_id",pl.col("s1_id").alias("ts1"),pl.col("pr").alias("tpr"))
x=x.join(top,on="s23_id",how="left").with_columns((pl.col("ts1")==pl.col("s1_id")).alias("istop"))
print(x.group_by("keq",pl.col("pr").is_null().alias("notret"),"istop").agg(pl.len(),pl.col("pr").median(),pl.col("tpr").median()).sort("keq","notret"))
