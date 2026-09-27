import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W = "../work/"
T = {"US": .75, "India": .8, "France": .75}
# test: queries per country (by top-1 S1 country), accepted share, and top-1 p2 distribution of the unaccepted
sc = pl.read_parquet(W + "test_scores_stage2_v11.parquet").join(pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id")
top = sc.sort("p2", descending=True).unique("s23_id", keep="first")
m = pl.read_csv(W + "probe11_fr_vfix/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).rename({"source1_entity_id": "s1_id"})
acc = m.filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(",").alias("s23_id")).explode("s23_id").select("s23_id", pl.lit(True).alias("acc"))
top = top.join(acc, on="s23_id", how="left").with_columns(pl.col("acc").fill_null(False))
ns1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["country"]).group_by("country").len("ns1")
print("TEST per 1000 S1")
print(top.group_by("country").agg((pl.len()).alias("q"), pl.col("acc").sum().alias("a"),
      (~pl.col("acc") & (pl.col("p") >= .5)).sum().alias("rej_p.5"), (~pl.col("acc") & (pl.col("p") >= .1) & (pl.col("p") < .5)).sum().alias("rej_p.1-.5"))
      .join(ns1, on="country").with_columns([(pl.col(c) / pl.col("ns1") * 1000).round(1) for c in ("q", "a", "rej_p.5", "rej_p.1-.5")]).sort("country"))
# total queries in S2+S3 not in any candidate pair
allq = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id"]) for i in (2, 3)])
print("test queries total", allq.height, "with candidates", top.height)
v = pl.read_parquet(W + "valid_scores_stage2.parquet").join(pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id").filter(is_valid_expr("s1_id"))
vt = v.sort("p2", descending=True).unique("s23_id", keep="first").with_columns((pl.col("p2") >= pl.col("country").replace_strict(T)).alias("acc"))
nv = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).group_by("country").len("ns1")
print("VALID per 1000 S1")
print(vt.group_by("country").agg(pl.len().alias("q"), pl.col("acc").sum().alias("a"), pl.col("label").sum().alias("true_top1"),
      (~pl.col("acc") & (pl.col("p") >= .5)).sum().alias("rej_p.5"), (~pl.col("acc") & (pl.col("p") >= .1) & (pl.col("p") < .5)).sum().alias("rej_p.1-.5"))
      .join(nv, on="country").with_columns([(pl.col(c) / pl.col("ns1") * 1000).round(1) for c in ("q", "a", "true_top1", "rej_p.5", "rej_p.1-.5")]).sort("country"))
