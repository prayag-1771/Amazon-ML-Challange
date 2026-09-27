import polars as pl
pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40)
from src.io_utils import load_source, load_ground_truth
gt=load_ground_truth()
s1=load_source("train",1).select(pl.col("entity_id").alias("s1_id"),"country")
g=gt.with_columns(pl.col("s23_id").str.slice(0,2).alias("src"))
per=s1.join(g.group_by("s1_id").agg((pl.col("src")=="S2").sum().alias("n2"),(pl.col("src")=="S3").sum().alias("n3")),on="s1_id",how="left").fill_null(0)
per=per.with_columns((pl.col("n2")+pl.col("n3")).alias("n"))
print(per.group_by("country","n").len().sort("country","n").pivot(on="country",index="n",values="len"))
print(per.group_by("n2","n3").len().sort("len",descending=True).head(25))
