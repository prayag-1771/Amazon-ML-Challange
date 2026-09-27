"""Accepted-pair categories by name/number relation: France test vs US/India val (with precision)."""
import sys
sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W = "../work/"
def cats(x):
    t1 = pl.col("n1").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    tq = pl.col("nq").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    x = x.with_columns(tq.list.set_difference(t1).list.len().alias("add"), t1.list.set_difference(tq).list.len().alias("drop"),
                       pl.col("u1").str.split(" ").list.first().alias("a1"), pl.col("uq").str.split(" ").list.first().alias("aq"))
    num = (pl.when(pl.col("aq").fill_null("") == "").then(pl.lit("qnonum")).when(pl.col("a1") == pl.col("aq")).then(pl.lit("num=")).otherwise(pl.lit("num!=")))
    nm = (pl.when(pl.col("n1") == pl.col("nq")).then(pl.lit("name="))
          .when((pl.col("add") == 0) & (pl.col("drop") == 0)).then(pl.lit("reorder"))
          .when((pl.col("add") == 1) & (pl.col("drop") == 1)).then(pl.lit("swap1"))
          .when((pl.col("add") >= 1) & (pl.col("drop") == 0)).then(pl.lit("addonly"))
          .when((pl.col("add") == 0) & (pl.col("drop") >= 1)).then(pl.lit("droponly"))
          .otherwise(pl.lit("other")))
    return x.with_columns(nm.alias("nm"), num.alias("num"))
def load(split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums", "state"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq", "state": "stq"})
    return s1, q
s1, q = load("test")
d = pl.read_parquet(W + "test_scores_stage2_v9.parquet")
top = d.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
top = cats(top.filter(pl.col("country") == "France"))
T = 0.75
acc = top.filter(pl.col("p2") >= T)
n = acc.height
print("FRANCE test accepted", n)
print(acc.group_by("nm", "num").agg(pl.len().alias("n"), (pl.len() / n).round(4).alias("share"), pl.col("p2").mean().round(4).alias("p"),
      (pl.col("p2") < 0.999).mean().round(3).alias("band")).sort("n", descending=True))
band = top.filter(pl.col("p2").is_between(0.02, 0.75))
print("FRANCE rejected band 0.02-0.75", band.height)
print(band.group_by("nm", "num").agg(pl.len().alias("n")).sort("n", descending=True).head(10))
s1, q = load("train")
a = pl.read_parquet(W + "valid_scores_lgb_v5cf.parquet").rename({"p": "pa"})
b = pl.read_parquet(W + "valid_scores_lgb_v6k.parquet")
v = a.join(b, on=["s1_id", "s23_id"]).with_columns(((pl.col("pa") + pl.col("p")) / 2).alias("p2"))
top = cats(v.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id"))
for c in ("US", "India"):
    x = top.filter((pl.col("country") == c) & (pl.col("p2") >= T)); n = x.height
    print(c, "VAL accepted", n)
    print(x.group_by("nm", "num").agg(pl.len().alias("n"), (pl.len() / n).round(4).alias("share"), pl.col("label").mean().round(4).alias("prec"),
          (pl.col("p2") < 0.999).mean().round(3).alias("band")).sort("n", descending=True))
    y = top.filter((pl.col("country") == c) & (pl.col("p2") < T) & (pl.col("label") == 1))
    print(c, "VAL missed-but-top1 true", y.height, y.group_by("nm", "num").len().sort("len", descending=True).head(6).rows())
