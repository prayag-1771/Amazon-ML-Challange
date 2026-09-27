"""House-number string edit type between S1 and query first number, top-1 pairs with name= or var/typo swap:
val precision (US/India) vs France acceptance (vfix)."""
import sys
sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W = "../work/"
src = open("../work/v10/frcat.py").read()
exec(src[src.index("def cats"):src.index("def load")])
def etype(a, b):
    if not a or not b: return "missing"
    if a == b: return "eq"
    try:
        d = int(b) - int(a)
    except ValueError:
        return "nonint"
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]: return "transp"
        if len(diff) == 1: return f"sub{'±1' if abs(d) <= 2 else ''}{'+' if d > 0 else '-'}" if abs(d) <= 25 else "sub_far"
    if abs(len(a) - len(b)) == 1:
        s, l = (a, b) if len(a) < len(b) else (b, a)
        if any(l[:i] + l[i + 1:] == s for i in range(len(l))): return "drop" if len(b) < len(a) else "ins"
    if abs(d) <= 25: return "near+" if d > 0 else "near-"
    return "far"
def prep(split, sc):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = cats(sc.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id"))
    x = x.filter((pl.col("nm") == "name=") & (pl.col("num") != "num="))
    return x.with_columns(pl.struct("a1", "aq").map_elements(lambda s: etype(s["a1"], s["aq"]), return_dtype=pl.String).alias("e"))
v = prep("train", pl.read_parquet(W + "valid_scores_stage2.parquet"))
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id"))
for c in ("US", "India"):
    n = s1v.filter(pl.col("country") == c).height
    x = v.filter(pl.col("country") == c)
    print(c, x.group_by("e").agg((pl.len() / n * 1000).round(1).alias("perK"), pl.col("label").mean().round(3).alias("prec"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")).sort("perK", descending=True).rows())
t = prep("test", pl.read_parquet(W + "test_scores_stage2_v11.parquet"))
m = pl.read_csv(W + "probe11_fr_vfix/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).rename({"source1_entity_id": "s1_id"})
m = m.filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(",").alias("s23_id")).explode("s23_id").select("s1_id", "s23_id").with_columns(pl.lit(True).alias("acc"))
t = t.join(m, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("acc").fill_null(False))
nf = (pl.read_parquet(W + "norm_test_s1.parquet", columns=["country"])["country"] == "France").sum()
x = t.filter(pl.col("country") == "France")
print("FR", x.group_by("e").agg((pl.len() / nf * 1000).round(1).alias("perK"), pl.col("acc").mean().round(3).alias("acc"), pl.col("p2").mean().round(3).alias("p2")).sort("perK", descending=True).rows())
for e in ("transp", "drop", "ins", "sub-", "sub_far", "far"):
    print("  FR", e, x.filter((pl.col("e") == e) & ~pl.col("acc")).select("n1", "u1", "uq", "p2").head(6).rows())
