import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W="../work/"
gt=load_ground_truth()
s1tr=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
q=pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet",columns=["entity_id"]).with_columns(pl.lit(k).alias("src")) for k in (2,3)]).rename({"entity_id":"s23_id"})
print(gt.columns)
g=gt.join(q,on="s23_id").group_by("s1_id").agg((pl.col("src")==2).sum().alias("n2"),(pl.col("src")==3).sum().alias("n3"))
g=s1tr.join(g,on="s1_id",how="left").fill_null(0)
print("TRAIN true counts per S1:",g.group_by("country").agg(pl.col("n2").mean(),pl.col("n3").mean(),(pl.col("n2")+pl.col("n3")==0).mean().alias("single")))
print(g.group_by("country","n2").len().sort("country","n2").pivot(on="country",index="n2",values="len").head(8))
print(g.group_by("country","n3").len().sort("country","n3").pivot(on="country",index="n3",values="len").head(8))
# test predictions
s1t=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
qt=pl.concat([pl.read_parquet(W+f"norm_test_s{k}.parquet",columns=["entity_id"]).with_columns(pl.lit(k).alias("src")) for k in (2,3)]).rename({"entity_id":"s23_id"})
print("test S1 by country",s1t.group_by("country").len(), "train", s1tr.group_by("country").len(), "test q",len(qt),"train q",len(q))
d=pl.read_parquet(W+"test_scores_ens_v8.parquet")
top=d.sort("p",descending=True).unique("s23_id",keep="first").join(qt,on="s23_id")
for t in [0.5,0.7,0.8,0.9]:
    a=top.filter(pl.col("p")>=t).group_by("s1_id").agg((pl.col("src")==2).sum().alias("n2"),(pl.col("src")==3).sum().alias("n3"))
    a=s1t.join(a,on="s1_id",how="left").fill_null(0)
    print("TEST t",t,a.group_by("country").agg(pl.col("n2").mean(),pl.col("n3").mean(),(pl.col("n2")+pl.col("n3")==0).mean().alias("single")).sort("country"))
