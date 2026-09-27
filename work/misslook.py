import sys; sys.path.insert(0, ".")
import polars as pl
from src.io_utils import load_ground_truth
from src.config import is_valid_expr
W="../work/"
gt=load_ground_truth()
s1=pl.read_parquet(W+"norm_train_s1.parquet").filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
print(s1.columns)
q=pl.concat([pl.read_parquet(W+f"norm_train_s{i}.parquet") for i in (2,3)]).rename({"entity_id":"s23_id"})
v=pl.read_parquet(W+"valid_scores_stage2.parquet",columns=["s1_id","s23_id"])
g=gt.join(s1.select("s1_id","country"),on="s1_id")
miss=g.join(v,on=["s1_id","s23_id"],how="anti")
# does the query have any candidate?
hasc=v.select("s23_id").unique().with_columns(pl.lit(True).alias("hasc"))
miss=miss.join(hasc,on="s23_id",how="left").with_columns(pl.col("hasc").fill_null(False))
miss=miss.join(q.select("s23_id",pl.col("addr_core").alias("aq"),pl.col("name_core").alias("nq"),pl.col("nums").alias("uq")),on="s23_id").join(s1.select("s1_id",pl.col("addr_core").alias("a1"),pl.col("name_core").alias("n1"),pl.col("nums").alias("u1")),on="s1_id")
miss=miss.with_columns((pl.col("aq").fill_null("")=="").alias("qe"),(pl.col("n1")==pl.col("nq")).alias("neq"),(pl.col("u1").str.split(" ").list.first()==pl.col("uq").str.split(" ").list.first()).fill_null(False).alias("numeq"))
print(miss.group_by("country","qe","hasc","neq","numeq").len().sort("len",descending=True).head(20))
for c in ("India","US"):
    print("==",c)
    for r in miss.filter((pl.col("country")==c)&~pl.col("qe")).sample(20,seed=2).select("hasc","n1","a1","nq","aq").rows(): print("  ",r)
miss.write_parquet(W+"miss_v12.parquet")
