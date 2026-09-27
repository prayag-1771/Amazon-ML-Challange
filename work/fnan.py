import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
T={"US":.75,"India":.8}
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
print("queries w/ >1 true s1:", gt.group_by("s23_id").len().filter(pl.col("len")>1).height, "of", gt["s23_id"].n_unique())
v=pl.read_parquet("valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first")
acc=top.filter(pl.col("p2")>=pl.col("country").replace_strict(T)).select("s1_id","s23_id")
fn=gt.join(acc,on=["s1_id","s23_id"],how="anti")
fn=fn.join(v.select("s1_id","s23_id","p2"),on=["s1_id","s23_id"],how="left")
fn=fn.join(top.select("s23_id",pl.col("s1_id").alias("top_s1"),pl.col("p2").alias("top_p2")),on="s23_id",how="left")
fn=fn.with_columns(pl.when(pl.col("p2").is_null()).then(pl.lit("nocand")).when(pl.col("top_s1")!=pl.col("s1_id")).then(pl.lit("lost_top1")).otherwise(pl.lit("below_T")).alias("kind"))
print(fn.group_by("kind").len())
lt=fn.filter(pl.col("kind")=="lost_top1")
print("lost_top1: top_s1 is true too?", lt.join(gt.rename({"s1_id":"top_s1"}),on=["top_s1","s23_id"],how="semi").height)
print(lt.select(pl.col("p2").describe() if False else pl.col("p2").quantile(q).alias(str(q)) for q in (.1,.25,.5,.75,.9)))
print(lt.select(pl.col("top_p2").quantile(q).alias(str(q)) for q in (.1,.25,.5,.75,.9)))
b=fn.filter(pl.col("kind")=="below_T")
print(b.select(pl.col("p2").quantile(q).alias(str(q)) for q in (.1,.25,.5,.75,.9)))
# second-best p2 distribution among lost_top1 relative to T
print("lost_top1 with p2>=0.5:", lt.filter(pl.col("p2")>=.5).height, ">=.3", lt.filter(pl.col("p2")>=.3).height)
