"""Empty-address queries: does an accepted (address-bearing) sibling query of the candidate S1 share the query's name?"""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W="../work/"; T={"US":.75,"India":.8}
s1=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
v=pl.read_parquet(W+"valid_scores_stage2.parquet").join(s1,on="s1_id")
q=pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet",columns=["entity_id","business_name","name_core","name_full","addr_empty"]).with_columns(pl.lit(k).alias("src")) for k in (2,3)]).rename({"entity_id":"s23_id"})
top=v.sort("p2",descending=True).unique("s23_id",keep="first").join(q,on="s23_id")
th=pl.col("country").replace_strict(T)
acc=top.filter(pl.col("p2")>=th).select("s1_id",pl.col("s23_id").alias("sib"),pl.col("business_name").alias("bn"),pl.col("name_full").alias("nf"),pl.col("src").alias("ssrc"),pl.col("addr_empty").alias("sempty"))
# all candidates of empty queries in the ambiguous band
c=v.join(q,on="s23_id").filter((pl.col("addr_empty")==1))
c=c.with_columns(pl.col("p2").rank("ordinal",descending=True).over("s23_id").alias("rk"))
j=c.join(acc,on="s1_id",how="left").filter(pl.col("sib").is_null()|(pl.col("sib")!=pl.col("s23_id")))
g=j.group_by("s1_id","s23_id").agg(pl.col("label").first(),pl.col("p2").first(),pl.col("rk").first(),pl.col("country").first(),
    (pl.col("bn")==pl.col("business_name")).any().alias("raw_eq"),(pl.col("nf")==pl.col("name_full")).any().alias("full_eq"),
    ((pl.col("bn")==pl.col("business_name"))&(pl.col("ssrc")!=pl.col("src"))).any().alias("raw_eq_x"),
    pl.col("sib").drop_nulls().len().alias("nsib"))
b=g.filter(pl.col("p2").is_between(.2,.9))
print(b.group_by("country","raw_eq","full_eq").agg(pl.len(),pl.col("label").mean().round(3),(pl.col("rk")==1).mean().round(2).alias("top1")).sort("country","raw_eq","full_eq").rows())
print(b.group_by("country","raw_eq_x").agg(pl.len(),pl.col("label").mean().round(3)).sort("country","raw_eq_x").rows())
print(b.group_by("country",pl.col("nsib").clip(upper_bound=5)).agg(pl.len(),pl.col("label").mean().round(3)).sort("country","nsib").rows())
