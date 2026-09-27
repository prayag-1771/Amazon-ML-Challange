import sys, polars as pl
from src.blocking import build_candidates, with_ids
from src.io_utils import load_ground_truth
from src.config import is_valid_expr
from src.normalize import normalize_source
split = "train"
cand = with_ids(build_candidates(split), split)
gt = load_ground_truth()
s1 = normalize_source("train", 1).select(pl.col("entity_id").alias("s1_id"), "country")
gt = gt.join(s1, on="s1_id").with_columns(is_valid_expr().alias("v"))
hit = gt.join(cand.select("s1_id", "s23_id", "emb_sim", "tf_sim").with_columns(pl.lit(True).alias("hit")),
              on=["s1_id", "s23_id"], how="left").with_columns(pl.col("hit").fill_null(False))
print(f"cands {len(cand):,}, per query {len(cand)/cand['q_row'].n_unique():.2f}")
print(hit.group_by("country", "v").agg(pl.col("hit").mean().alias("union"),
      pl.col("emb_sim").is_not_null().mean().alias("emb"), pl.col("tf_sim").is_not_null().mean().alias("tf"),
      pl.len()).sort("country", "v"))
