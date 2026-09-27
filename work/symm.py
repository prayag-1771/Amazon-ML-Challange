import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W = "../work/"
T = {"US": 0.75, "India": 0.80, "France": 0.75}
def prep(sc, split, filt=None):
    c = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "legal", "name_core", "nums"]).rename({"entity_id": "s1_id", "legal": "l1", "name_core": "n1", "nums": "u1"})
    if filt is not None: c = c.filter(filt)
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "legal", "name_core", "nums", "addr_empty"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "legal": "lq", "name_core": "nq", "nums": "uq"})
    d = pl.read_parquet(W + sc)
    top = d.sort("p2", descending=True).unique("s23_id", keep="first").filter(pl.col("p2") > 0.02).join(c, on="s1_id").join(q, on="s23_id")
    a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False); b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
    top = top.with_columns((b - a).alias("dif"),
        pl.when(pl.col("l1").fill_null("") == pl.col("lq").fill_null("")).then(pl.lit("same")).when(pl.col("l1").fill_null("") == "").then(pl.lit("add")).when(pl.col("lq").fill_null("") == "").then(pl.lit("drop")).otherwise(pl.lit("chg")).alias("leg"),
        (pl.col("n1") == pl.col("nq")).alias("nsame"), (pl.col("p2") >= pl.col("country").replace_strict(T)).alias("acc"))
    return top.filter(pl.col("dif").abs().is_between(1, 25))
v = prep("valid_scores_stage2.parquet", "train", is_valid_expr("s1_id"))
t = prep("test_scores_stage2_v10.parquet", "test")
K = ["country", "leg", "nsame"]
def tab(x, lab):
    aggs = [pl.len().alias("n"), pl.col("acc").sum().alias("acc")] + ([pl.col("label").sum().alias("pos")] if lab else [])
    g = x.group_by(K + [(pl.col("dif") > 0).alias("up")]).agg(aggs)
    up = g.filter(pl.col("up")).drop("up"); dn = g.filter(~pl.col("up")).drop("up")
    return up.join(dn, on=K, suffix="_dn", how="full", coalesce=True).sort(K)
print("VAL"); [print("  ", r) for r in tab(v, True).rows()]
print("TEST"); [print("  ", r) for r in tab(t, False).rows()]
print("VAL by |d| (US, all):", v.filter(pl.col("country") == "US").group_by(pl.col("dif")).agg(pl.len(), pl.col("label").mean().round(2)).sort("dif").rows())
print("TEST by d (US):", t.filter(pl.col("country") == "US").group_by(pl.col("dif")).agg(pl.len(), pl.col("acc").mean().round(2)).sort("dif").rows())
print("TEST by d (France):", t.filter(pl.col("country") == "France").group_by(pl.col("dif")).agg(pl.len(), pl.col("acc").mean().round(2)).sort("dif").rows())
