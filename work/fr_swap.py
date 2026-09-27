import sys; sys.path.insert(0, ".")
import polars as pl
from src.stage2 import struct_feats
W = "../work/"
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(250)
def load(split, sc):
    d = pl.read_parquet(sc)
    top = d.sort("p2", descending=True).unique("s23_id", keep="first")
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "business_name", "business_address"]).rename({"entity_id": "s1_id", "name_core": "n1", "business_name": "bn1", "business_address": "ba1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "business_name", "business_address"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "business_name": "bnq", "business_address": "baq"})
    top = struct_feats(top, split).join(s1, on="s1_id").join(q, on="s23_id")
    t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
    return top.with_columns(tq.list.set_difference(t1).list.first().alias("a"), t1.list.set_difference(tq).list.first().alias("d"))
v = load("train", W + "valid_scores_stage2.parquet")
sw = (pl.col("add_n") == 1) & (pl.col("drop_n") == 1) & (pl.col("nrel_c") == 0)
print("US val swap1-eq negatives"); print(v.filter(sw & (pl.col("country") == "US") & (pl.col("label") == 0)).sample(20, seed=2).select(pl.col("p2").round(2), "bn1", "bnq", "ba1", "baq"))
print("US val swap1-eq positives"); print(v.filter(sw & (pl.col("country") == "US") & (pl.col("label") == 1)).sample(15, seed=2).select(pl.col("p2").round(2), "bn1", "bnq", "ba1", "baq"))
t = load("test", W + "test_scores_stage2_v10.parquet")
f = t.filter(sw & (pl.col("country") == "France"))
print("FR swap1-eq rejected"); print(f.filter(pl.col("p2") < 0.75).sample(30, seed=3).select(pl.col("p2").round(2), "bn1", "bnq", "ba1", "baq"))
print(f.group_by("d", "a").agg(pl.len(), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")).sort("len", descending=True).head(40))
