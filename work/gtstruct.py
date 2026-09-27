import sys; sys.path.insert(0, ".")
import polars as pl
from src.io_utils import load_ground_truth
W="../work/"
gt=load_ground_truth()
s2=pl.read_parquet(W+"norm_train_s2.parquet",columns=["entity_id"]).with_columns(pl.lit(2).alias("src"))
s3=pl.read_parquet(W+"norm_train_s3.parquet",columns=["entity_id"]).with_columns(pl.lit(3).alias("src"))
q=pl.concat([s2,s3]).rename({"entity_id":"s23_id"})
c=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
g=gt.join(q,on="s23_id").join(c,on="s1_id")
k=c.join(g.group_by("s1_id").agg((pl.col("src")==2).sum().alias("n2"),(pl.col("src")==3).sum().alias("n3")),on="s1_id",how="left").fill_null(0)
for cc in ("US","India"):
    x=k.filter(pl.col("country")==cc)
    print(cc,"n2 dist",x.group_by("n2").len().sort("n2").head(12).rows())
    print(cc,"n3 dist",x.group_by("n3").len().sort("n3").head(12).rows())
    print(cc,"tot dist",x.group_by((pl.col("n2")+pl.col("n3")).alias("t")).len().sort("t").head(15).rows())
print("queries s2,s3", s2.height, s3.height, "matched", g.group_by("src").len().rows())
t2=pl.read_parquet(W+"norm_test_s2.parquet",columns=["entity_id"]).height; t3=pl.read_parquet(W+"norm_test_s3.parquet",columns=["entity_id"]).height
print("test s2,s3", t2, t3)
