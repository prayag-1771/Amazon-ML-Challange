import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
pl.Config.set_tbl_rows(60)
s1n=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country","name_core","addr_empty"]).rename({"entity_id":"s1_id","name_core":"n1","addr_empty":"s1e"})
s1n=s1n.join(s1n.group_by("country","n1").len("s1_ncnt"),on=["country","n1"])
s1=s1n.filter(is_valid_expr("s1_id")).select("s1_id","country")
gtall=load_ground_truth(); gt=gtall.join(s1.select("s1_id"),on="s1_id",how="semi")
pred=pl.read_parquet("pred15_v.parquet")
v=pl.read_parquet("valid_scores_stage2.parquet")
vt=v.sort("p2",descending=True).unique("s23_id",keep="first").select("s23_id",pl.col("s1_id").alias("vs1"),pl.col("p2").alias("vp2"))
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','country','name_core','addr_empty']).with_columns(pl.lit(k).alias("src")) for k in (2,3)]).rename({'entity_id':'s23_id'})
qv=q.join(gtall.join(s1.select("s1_id"),on="s1_id",how="anti").select("s23_id"),on="s23_id",how="anti")
un=qv.filter(pl.col("addr_empty")==1).join(pred.select("s23_id"),on="s23_id",how="anti")
acc=pred.join(q,on="s23_id")
fam=acc.group_by("country","name_core").agg(pl.col("s1_id").unique().alias("s1s"),pl.len().alias("nacc"),(pl.col("addr_empty")==1).sum().alias("nacc_e"))
fam=fam.filter(pl.col("s1s").list.len()==1).with_columns(pl.col("s1s").list.first().alias("s1_id")).drop("s1s")
c=un.join(fam,on=["country","name_core"]).join(gt.with_columns(pl.lit(1).alias("y")),on=["s1_id","s23_id"],how="left").fill_null(0)
c=c.join(s1n.select("s1_id","n1","s1e","s1_ncnt"),on="s1_id").join(vt,on="s23_id",how="left")
c=c.with_columns((pl.col("n1")==pl.col("name_core")).alias("s1eq"),pl.len().over("country","name_core").alias("nun"),
   (pl.col("vs1")==pl.col("s1_id")).fill_null(False).alias("topagree"),pl.col("vp2").is_null().alias("nocand"))
for g in (["s1eq"],["topagree","nocand"],["s1_ncnt"],["nun"],["nacc"]):
    print(c.group_by(g).agg(pl.len(),pl.col("y").mean().round(3)).sort(g))
c.write_parquet("trans_c.parquet")
m0=macro_f05(s1["s1_id"],pred,gt,by=s1)
def ev(name,a):
    m=macro_f05(s1["s1_id"],pl.concat([pred,a.select("s1_id","s23_id",pl.lit("tr").alias("ch"))]),gt,by=s1)
    print(name,a.height,round(a["y"].mean(),3),round(m["f05"]-m0["f05"],5),round(m["f05_US"]-m0["f05_US"],5),round(m["f05_India"]-m0["f05_India"],5))
ev("all",c)
ev("s1eq",c.filter("s1eq"))
ev("s1eq&nun1",c.filter(pl.col("s1eq")&(pl.col("nun")==1)))
ev("s1eq&ncnt1",c.filter(pl.col("s1eq")&(pl.col("s1_ncnt")==1)))
ev("topagree",c.filter("topagree"))
