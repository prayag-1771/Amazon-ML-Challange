import sys; sys.path.insert(0, ".")
import polars as pl
import src.stage2 as S
from src.config import is_valid_expr
W="../work/"
v=pl.read_parquet(W+"valid_scores_stage2.parquet")
c=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
top=v.sort("p2",descending=True).unique("s23_id",keep="first").join(c,on="s1_id")
t2=S.legal_add_rule(top.with_columns(pl.col("p2").alias("p2o")),v,"train")
th=pl.col("country").replace_strict(S.T,default=0.75)
print("val changed", t2.filter((pl.col("p2")>=th)&(pl.col("p2o")<th)).group_by("country").agg(pl.len(),pl.col("label").mean()).rows())
