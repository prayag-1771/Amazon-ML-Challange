"""Per-S1 rates of single-token swap/add at top-1, by added token and number relation: val all candidates' top-1
(US/IN, with label) vs France test. Generator vocabularies should show equal per-S1 rates across countries."""
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
    x = x.filter(pl.col("nm").is_in(["swap1", "addonly"]))
    t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
    x = x.with_columns(tq.list.set_difference(t1).alias("al"), t1.list.set_difference(tq).list.first().alias("dt"))
    return x.filter(pl.col("al").list.len() == 1).with_columns(pl.col("al").list.first().alias("at"))
nv = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).group_by("country").len("ns")
nt = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).group_by("country").len("ns")
print(nv.rows(), nt.rows())
v = prep("train", pl.read_parquet(W + "valid_scores_stage2.parquet")).join(nv, on="country")
t = prep("test", pl.read_parquet(W + "test_scores_stage2_v11.parquet")).join(pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id").join(nt, on="country")
def tab(x, keys, lab):
    a = [pl.len().alias("n"), (pl.len() / pl.col("ns").first() * 1000).round(2).alias("perK")]
    a += [pl.col("label").mean().round(3).alias("prec")] if lab else [(pl.col("p2") >= 0.75).mean().round(3).alias("acc")]
    return x.group_by(keys).agg(a).sort("n", descending=True)
for c in ("US", "India"):
    x = v.filter(pl.col("country") == c)
    print("\n==", c)
    for nm in ("swap1", "addonly"):
        for num in ("num=", "num!="):
            print(nm, num, tab(x.filter((pl.col("nm") == nm) & (pl.col("num") == num)), "at", True).head(14).select("at", "perK", "prec").rows())
print("\n== France test (v11)")
x = t.filter(pl.col("country") == "France")
for nm in ("swap1", "addonly"):
    for num in ("num=", "num!="):
        print(nm, num, tab(x.filter((pl.col("nm") == nm) & (pl.col("num") == num)), "at", False).head(16).select("at", "perK", "acc").rows())
