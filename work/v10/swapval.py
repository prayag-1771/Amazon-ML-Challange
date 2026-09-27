"""Val (US/India): precision of equal-number single-token swaps by (dropped, added) token pair. Is a true-match swap
a fixed synonym pair (directed map) while distractor swaps are arbitrary generic words?"""
import sys
sys.path.insert(0, ".")
import polars as pl
from rapidfuzz.distance import Levenshtein as L
from src.config import is_valid_expr
W = "../work/"
src = open("../work/v10/frcat.py").read()
exec(src[src.index("def cats"):src.index("def load")])
def prep(split, scores):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = cats(scores.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id"))
    x = x.filter((pl.col("nm") == "swap1") & (pl.col("num") == "num="))
    t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
    x = x.with_columns(t1.list.set_difference(tq).list.first().alias("dt"), tq.list.set_difference(t1).list.first().alias("at"))
    return x.with_columns(pl.struct("dt", "at").map_elements(lambda s: L.normalized_similarity(s["dt"] or "", s["at"] or ""), return_dtype=pl.Float64).alias("sim")).filter(pl.col("sim") < 0.5)
v = prep("train", pl.read_parquet(W + "valid_scores_stage2.parquet"))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
for c in ("US", "India"):
    x = v.filter(pl.col("country") == c)
    print(c, "swap1 num= sim<.5 top1", x.height, "prec", round(x["label"].mean(), 4), "acc", round((x["p2"] >= 0.75).mean(), 3),
          "prec|acc", round(x.filter(pl.col("p2") >= 0.75)["label"].mean(), 4), "recall-miss", x.filter((pl.col("p2") < 0.75) & (pl.col("label") == 1)).height)
    g = x.group_by("dt", "at").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("prec"), pl.col("p2").mean().round(3).alias("p2"))
    print("  distinct pairs", g.height, "top pairs cover", g.sort("n", descending=True).head(30)["n"].sum())
    print(g.sort("n", descending=True).head(25))
    # directedness: for true swaps, is (dt->at) mostly one fixed pair per dt?
    t = x.filter(pl.col("label") == 1).group_by("dt").agg(pl.col("at").n_unique().alias("n_at"), pl.len().alias("n")).sort("n", descending=True).head(10)
    f = x.filter(pl.col("label") == 0).group_by("dt").agg(pl.col("at").n_unique().alias("n_at"), pl.len().alias("n")).sort("n", descending=True).head(10)
    print("  TRUE dt fanout", t.rows()); print("  FALSE dt fanout", f.rows())
t = prep("test", pl.read_parquet(W + "test_scores_stage2_v11.parquet").join(pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id"]).rename({"entity_id": "s1_id"}), on="s1_id"))
x = t.filter(pl.col("country") == "France")
print("FRANCE test swap1 num= sim<.5 top1", x.height, "acc", round((x["p2"] >= 0.75).mean(), 3))
g = x.group_by("dt", "at").agg(pl.len().alias("n"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc"), pl.col("p2").mean().round(3).alias("p2"))
print("  distinct pairs", g.height); print(g.sort("n", descending=True).head(40))
print("  dt fanout", x.group_by("dt").agg(pl.col("at").n_unique().alias("n_at"), pl.len().alias("n"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")).sort("n", descending=True).head(15).rows())
