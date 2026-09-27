"""How are distractors generated? Train queries with no true S1: compare with the closest S1 (top p0)."""
import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(70)
W = "../work/"
gt = pl.read_parquet(W + "gt_pairs.parquet")
pr = pl.read_parquet(W + "pruned_train.parquet", columns=["s1_id", "s23_id", "p0_qrank"]).filter(pl.col("p0_qrank") == 1)
dist = pr.join(gt, on="s23_id", how="anti")
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "business_name", "business_address", "name_full", "addr_core", "country"]).rename(
    {"entity_id": "s1_id", "business_name": "N1", "business_address": "A1", "name_full": "n1", "addr_core": "a1"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=["entity_id", "business_name", "business_address", "name_full", "addr_core"]) for k in (2, 3)]).rename(
    {"entity_id": "s23_id", "business_name": "NQ", "business_address": "AQ", "name_full": "nq", "addr_core": "aq"})
d = dist.join(s1, on="s1_id").join(q, on="s23_id").with_columns(
    pl.col("nq").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).alias("xq"),
    pl.col("n1").str.split(" ").list.set_difference(pl.col("nq").str.split(" ")).alias("x1"),
    (pl.col("a1") == pl.col("aq")).alias("same_addr"))
d = d.with_columns(pl.when(pl.col("xq").list.len() == 0).then(pl.lit("no_extra")).when(pl.col("x1").list.len() == 0).then(pl.lit("added")).otherwise(pl.lit("swap")).alias("kind"))
print(d.group_by("country", "kind").agg(pl.len(), pl.col("same_addr").mean()).sort("country", "kind"))
for k in ("added", "swap", "no_extra"):
    print("=====", k)
    for r in d.filter(pl.col("kind") == k).sample(8, seed=5).iter_rows(named=True):
        print(f"  S1: {r['N1']} | {r['A1']}\n  Q : {r['NQ']} | {r['AQ']}")
# France test: same analysis on top-1 with p in uncertain band and >0.999
