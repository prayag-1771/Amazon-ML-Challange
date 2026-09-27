"""Added-token ('at') vocabulary of equal-number single-token swaps: val true vs false (US/India), France test."""
import sys
sys.path.insert(0, ".")
import polars as pl
from rapidfuzz.distance import Levenshtein as L
W = "../work/"
src = open("../work/v10/frcat.py").read()
exec(src[src.index("def cats"):src.index("def load")])
def prep(split, scores, allnum=False):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = cats(scores.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id"))
    x = x.filter(pl.col("nm").is_in(["swap1", "addonly"]))
    t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
    x = x.with_columns(t1.list.set_difference(tq).list.first().alias("dt"), tq.list.set_difference(t1).list.first().alias("at"), tq.list.set_difference(t1).list.len().alias("na"))
    return x.with_columns(pl.struct("dt", "at").map_elements(lambda s: L.normalized_similarity(s["dt"] or "", s["at"] or ""), return_dtype=pl.Float64).alias("sim"))
v = prep("train", pl.read_parquet(W + "valid_scores_stage2.parquet"))
for c in ("US", "India"):
    for nm in ("swap1", "addonly"):
        x = v.filter((pl.col("country") == c) & (pl.col("nm") == nm) & (pl.col("sim") < 0.5) & (pl.col("na") == 1))
        for num in ("num=", "num!="):
            y = x.filter(pl.col("num") == num)
            g = y.group_by("at").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("prec"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")).sort("n", descending=True)
            print(f"{c} {nm} {num} n={y.height} prec={y['label'].mean():.3f}  top at:", [(a, n, p, ac) for a, n, p, ac in g.head(18).rows()])
t = prep("test", pl.read_parquet(W + "test_scores_stage2_v11.parquet"))
x = t.filter((pl.col("country") == "France") & (pl.col("sim") < 0.5) & (pl.col("na") == 1))
for nm in ("swap1", "addonly"):
    for num in ("num=", "num!="):
        y = x.filter((pl.col("nm") == nm) & (pl.col("num") == num))
        g = y.group_by("at").agg(pl.len().alias("n"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc"), pl.col("p2").mean().round(3).alias("p2")).sort("n", descending=True)
        print(f"FR {nm} {num} n={y.height} acc={(y['p2']>=0.75).mean():.3f}  top at:", g.head(22).rows())
