import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
T={"US":.75,"India":.8}
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
v=pl.read_parquet("valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first")
top=top.with_columns((pl.col("p2")>=pl.col("country").replace_strict(T)).alias("acc"))
nacc=top.filter("acc").group_by("s1_id").len("nacc")
bt=top.filter(~pl.col("acc")&(pl.col("p2")>=.3)).join(nacc,on="s1_id",how="left").fill_null(0)
bt=bt.with_columns(pl.col("p2").cut([.4,.5,.6,.7]).alias("bin"),pl.col("nacc").clip(0,3))
print(bt.group_by("country","bin","nacc").agg(pl.len(),pl.col("label").mean().round(3)).sort("country","bin","nacc").to_pandas().to_string())
