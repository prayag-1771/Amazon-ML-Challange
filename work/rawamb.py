import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250)
T={"US":.75,"India":.8}
s1=pl.read_parquet("norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','addr_empty']) for k in (2,3)]).rename({'entity_id':'s23_id'})
r1=pl.read_parquet("raw_train_s1.parquet").select(pl.col("entity_id").alias("s1_id"),pl.col("business_name").alias("rn1"),pl.col("business_address").alias("ra1"))
rq=pl.concat([pl.read_parquet(f"raw_train_s{k}.parquet") for k in (2,3)]).select(pl.col("entity_id").alias("s23_id"),pl.col("business_name").alias("rnq"))
v=pl.read_parquet("valid_scores_stage2.parquet").join(s1,on="s1_id").join(q,on="s23_id").filter(pl.col("addr_empty")==1)
v=v.join(v.group_by("s23_id").agg(pl.col("p2").max().alias("mx")),on="s23_id").filter(pl.col("mx").is_between(.3,.75))
v=v.filter(pl.col("p2")>=pl.col("mx")-.3).join(r1,on="s1_id").join(rq,on="s23_id")
v=v.with_columns((pl.col("rn1")==pl.col("rnq")).alias("rawEq"),(pl.col("rn1").str.to_lowercase()==pl.col("rnq").str.to_lowercase()).alias("ciEq"))
print(v.group_by("rawEq","ciEq").agg(pl.len(),pl.col("label").mean().round(3)))
print(v.sort("s23_id","p2",descending=[False,True]).select("s23_id","rnq","rn1","ra1","p2","label").head(40))
