import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W="../work/"; T={"US":.75,"India":.8}
s1=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1,on="s1_id")
q=pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet",columns=["entity_id","addr_empty"]) for k in (2,3)]).rename({"entity_id":"s23_id"})
gt=gt.join(q,on="s23_id",how="left")
v=pl.read_parquet(W+"valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first")
base=top.filter(pl.col("p2")>=pl.col("country").replace_strict(T)).select("s1_id","s23_id")
r=pl.read_parquet(W+"v10/rescue_valid.parquet").filter(pl.col("vs1")).sort("pr",descending=True).unique("s23_id",keep="first").filter(pl.col("pr")>=.9).join(v.select("s23_id").unique(),on="s23_id",how="anti")
casc=pl.read_parquet(W+"cascade_valid_p001.parquet").select("s23_id").unique()
b14=pl.concat([base,r.select("s1_id","s23_id")])
a=pl.read_parquet(W+"v10/ac2_valid_mid.parquet").sort("pr",descending=True).unique("q_row",keep="first").filter(pl.col("vs1")).join(casc,on="s23_id",how="anti").join(b14,on="s23_id",how="anti").filter(pl.col("pr")>=.9)
pred=pl.concat([b14,a.select("s1_id","s23_id")]).with_columns(pl.lit(1).alias("hit"))
fn=gt.join(pred,on=["s1_id","s23_id"],how="anti")
inv=v.select("s1_id","s23_id","p2").join(top.select("s23_id",pl.col("s1_id").alias("top1"),pl.col("p2").alias("ptop")),on="s23_id")
fn=fn.join(inv,on=["s1_id","s23_id"],how="left").with_columns(
  pl.when(pl.col("p2").is_null()).then(pl.lit("notcand")).when(pl.col("top1")!=pl.col("s1_id")).then(pl.lit("outranked")).otherwise(pl.lit("below_t")).alias("cat"))
fp=pred.join(gt.select("s1_id","s23_id"),on=["s1_id","s23_id"],how="anti").join(q,on="s23_id").join(s1,on="s1_id")
print("FN",len(fn),"FP",len(fp),"of",len(gt))
print(fn.group_by("country","addr_empty","cat").len().sort("country","addr_empty","cat"))
print(fp.group_by("country","addr_empty").len().sort("country","addr_empty"))
print(fn.filter(pl.col("cat")=="below_t").with_columns((pl.col("p2")*10).floor().alias("b")).group_by("country","b").len().sort("country","b"))
fn.write_parquet(W+"fn15.parquet")
