"""France swap1/num= top-1s: which (dropped -> added) token pairs, and their accept rate; US/India val with truth."""
import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=220)
from src.config import is_valid_expr
W = "../work/"
exec(open("../work/v10/frcat.py").read().split("def load")[0].split("W = \"../work/\"")[1])
def load(split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    return s1, q
def sw(x):
    t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
    x = cats(x).filter((pl.col("nm") == "swap1") & (pl.col("num") == "num="))
    x = x.with_columns(t1.list.set_difference(tq).list.first().alias("dt"), tq.list.set_difference(t1).list.first().alias("at"))
    from rapidfuzz.distance import Levenshtein as L
    return x.with_columns(pl.struct("dt", "at").map_elements(lambda s: L.normalized_similarity(s["dt"], s["at"]), return_dtype=pl.Float64).alias("sim"))
T = 0.75
s1, q = load("test")
d = pl.read_parquet(W + "test_scores_stage2_v9.parquet")
fr = sw(d.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id").filter(pl.col("country") == "France"))
fr = fr.with_columns((pl.col("sim") >= 0.5).alias("typo"))
print("FR by typo-like", fr.group_by("typo").agg(pl.len(), (pl.col("p2") >= T).mean().round(3).alias("acc")))
print(fr.filter(~pl.col("typo")).group_by("dt", "at").agg(pl.len(), (pl.col("p2") >= T).mean().round(3).alias("acc"), pl.col("p2").median().round(3).alias("med")).sort("len", descending=True).head(30))
print("FR dropped-token (non-typo) top", fr.filter(~pl.col("typo")).group_by("dt").agg(pl.len(), (pl.col("p2") >= T).mean().round(3).alias("acc")).sort("len", descending=True).head(15))
print("FR added-token (non-typo) top", fr.filter(~pl.col("typo")).group_by("at").agg(pl.len(), (pl.col("p2") >= T).mean().round(3).alias("acc")).sort("len", descending=True).head(15))
s1, q = load("train")
v = pl.read_parquet(W + "valid_scores_stage2.parquet")
vt = sw(v.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")).with_columns((pl.col("sim") >= 0.5).alias("typo"))
print("VAL by typo-like", vt.group_by("country", "typo").agg(pl.len(), (pl.col("p2") >= T).mean().round(3).alias("acc"), pl.col("label").mean().round(3).alias("true")).sort("country", "typo"))
print(vt.filter(~pl.col("typo")).group_by("country", "dt", "at").agg(pl.len(), pl.col("label").mean().round(3).alias("true")).sort("len", descending=True).head(20))
