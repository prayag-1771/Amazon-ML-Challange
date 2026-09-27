import sys; sys.path.insert(0, ".")
import polars as pl
from src.stage2 import struct_feats
W = "../work/"
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
def load(split, sc):
    d = pl.read_parquet(sc)
    top = d.sort("p2", descending=True).unique("s23_id", keep="first")
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    top = struct_feats(top, split).join(s1, on="s1_id").join(q, on="s23_id")
    a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False); b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
    return top.with_columns((b - a).alias("df"), (pl.col("n1") == pl.col("nq")).alias("same"))
v = load("train", W + "valid_scores_stage2.parquet")
t = load("test", W + "test_scores_stage2_v10.parquet")
bins = lambda: pl.col("df").cut([-100, -26, -11, -2, -1, 0, 1, 10, 25, 100], labels=["<-100", "-100..-26", "-25..-11", "-10..-2", "-1", "0", "1", "2..10", "11..25", "26..100", ">100"]).alias("b")
for nm, cond in (("same name", pl.col("same")), ("other", ~pl.col("same"))):
    print("=====", nm)
    x = v.filter(cond & pl.col("df").is_not_null()).group_by("country", bins()).agg(pl.len(), pl.col("label").mean().round(3).alias("pos"))
    y = t.filter(cond & pl.col("df").is_not_null()).group_by("country", bins()).agg(pl.len(), (pl.col("p2") >= .75).mean().round(3).alias("acc"))
    x = x.pivot(on="country", index="b", values=["len", "pos"])
    y = y.pivot(on="country", index="b", values=["len", "acc"])
    print(x.join(y, on="b", how="full", coalesce=True).sort("b"))
print("######## per-country counts, FR included")
for nm, cond in (("same name", pl.col("same")), ("other", ~pl.col("same"))):
    y = t.filter(cond & pl.col("df").is_not_null()).group_by("country", bins()).agg(pl.len(), (pl.col("p2") >= .75).mean().round(3).alias("acc"))
    y = y.with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("sh")).pivot(on="country", index="b", values=["sh", "acc"])
    print(nm); print(y.sort("b"))
