import polars as pl
from src.io_utils import load_ground_truth
from src.config import is_valid_expr
from src.normalize import normalize_source
cand = pl.read_parquet("../work/pruned_train.parquet", columns=["s1_id", "s23_id", "p0"])
gt = load_ground_truth()
s1 = normalize_source("train", 1).select(pl.col("entity_id").alias("s1_id"), "country")
gt = gt.join(s1, on="s1_id").with_columns(is_valid_expr().alias("v"))
hit = gt.join(cand.with_columns(pl.lit(True).alias("hit")), on=["s1_id", "s23_id"], how="left") \
        .with_columns(pl.col("hit").fill_null(False))
print(f"pruned cands {len(cand):,}, per query {len(cand)/cand['s23_id'].n_unique():.2f}")
print(hit.group_by("country", "v").agg(pl.col("hit").mean().alias("recall"), pl.len()).sort("country", "v"))
