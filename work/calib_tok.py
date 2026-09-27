import sys; sys.path.insert(0, ".")
import polars as pl
from src.stage2 import struct_feats
W = "../work/"
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(260)
def load(split, sc):
    d = pl.read_parquet(sc)
    top = d.sort("p2", descending=True).unique("s23_id", keep="first")
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "business_name", "business_address"]).rename({"entity_id": "s1_id", "name_core": "n1", "business_name": "bn1", "business_address": "ba1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "business_name", "business_address"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "business_name": "bnq", "business_address": "baq"})
    top = struct_feats(top, split).join(s1, on="s1_id").join(q, on="s23_id")
    t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
    return top.with_columns(tq.list.set_difference(t1).list.first().alias("a"), t1.list.set_difference(tq).list.first().alias("d"))
mt = pl.read_parquet(W + "mixtab_train.parquet").filter(pl.col("side") == "add").select("country", pl.col("tok").alias("a"), "est")
v = load("train", W + "valid_scores_stage2.parquet").join(mt, on=["country", "a"], how="left")
c = (pl.col("add_n") == 1) & (pl.col("drop_n") == 0) & (pl.col("nrel_c") == 0)
g = v.filter(c).group_by("country", "a").agg(pl.len(), pl.col("label").mean().round(3).alias("pos"), (pl.col("p2") >= .75).mean().round(3).alias("acc"), pl.col("est").first().round(3)).filter(pl.col("len") >= 40)
print(g.filter(pl.col("est") < 0.97).sort("len", descending=True).head(40))
print("weighted", g.group_by("country", (pl.col("est") * 10).floor().alias("eb")).agg(pl.col("len").sum(), ((pl.col("pos") * pl.col("len")).sum() / pl.col("len").sum()).round(3).alias("pos"), ((pl.col("est") * pl.col("len")).sum() / pl.col("len").sum()).round(3).alias("est")).sort("country", "eb"))
sf = (pl.col("n1") == pl.col("nq")) & (pl.col("nrel_c") == 3)
print("US same-name far, positives"); print(v.filter(sf & (pl.col("country") == "US") & (pl.col("label") == 1)).sample(15, seed=1).select(pl.col("p2").round(2), "bn1", "ba1", "baq"))
print("US same-name far, negatives"); print(v.filter(sf & (pl.col("country") == "US") & (pl.col("label") == 0)).sample(15, seed=1).select(pl.col("p2").round(2), "bn1", "ba1", "baq"))
