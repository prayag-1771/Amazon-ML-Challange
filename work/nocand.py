import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
s1=pl.read_parquet("norm_train_s1.parquet").filter(is_valid_expr("entity_id"))
gt=load_ground_truth().join(s1.select(pl.col("entity_id").alias("s1_id")),on="s1_id",how="semi")
v=pl.read_parquet("valid_scores_stage2.parquet")
casc=pl.read_parquet("cascade_valid_p001.parquet")
pr=pl.read_parquet("pruned_train.parquet",columns=["s1_id","s23_id"])
print(casc.columns)
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet').with_columns(pl.lit(k).alias("src")) for k in (2,3)])
print(q.columns)
fn=gt.join(v.select("s1_id","s23_id"),on=["s1_id","s23_id"],how="anti")
fn=fn.with_columns(pl.col("s23_id").is_in(v["s23_id"].implode()).alias("q_has_cand"),
   pl.struct("s1_id","s23_id").is_in(pr.select(pl.struct("s1_id","s23_id")).to_series().implode()).alias("in_pruned"))
fn=fn.join(q.select(pl.col("entity_id").alias("s23_id"),"country","addr_empty","src"),on="s23_id")
fn=fn.join(s1.select(pl.col("entity_id").alias("s1_id"),pl.col("addr_empty").alias("s1_empty")),on="s1_id")
print(fn.group_by("country","addr_empty","q_has_cand","in_pruned").len().sort("len",descending=True))
fn.write_parquet("nocand_v.parquet")
