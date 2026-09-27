"""Hypothesis: a true variant swaps/adds a word from a small generic vocabulary (US/IN: center/services/service/partners);
an equal-number swap to any other (content) word is a same-address distractor. Check on val labels, size it on France test."""
import sys
sys.path.insert(0, ".")
import polars as pl
from rapidfuzz.distance import Levenshtein as L
W = "../work/"
src = open("../work/v10/frcat.py").read()
exec(src[src.index("def cats"):src.index("def load")])
V = ["center", "services", "service", "partners"]
GFR = ["fils", "services", "groupe", "developpement", "associes", "france", "compagnie", "cie", "service", "center", "centre", "partners"]
def prep(split, scores):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = cats(scores.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id"))
    x = x.filter(pl.col("nm").is_in(["swap1"]) & (pl.col("num") == "num="))
    t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
    x = x.with_columns(t1.list.set_difference(tq).list.first().alias("dt"), tq.list.set_difference(t1).list.first().alias("at"))
    x = x.with_columns(pl.struct("dt", "at").map_elements(lambda s: L.normalized_similarity(s["dt"] or "", s["at"] or ""), return_dtype=pl.Float64).alias("sim"))
    x = x.filter(pl.col("sim") < 0.5)
    ocr = pl.col("at").str.contains(r"\d") | (pl.col("at").str.len_chars() <= 2)
    return x.with_columns(pl.when(ocr).then(pl.lit("ocr")).when(pl.col("at").is_in(V)).then(pl.lit("V")).otherwise(pl.lit("content")).alias("k"),
                          pl.col("dt").is_in(V).alias("dtV"))
v = prep("train", pl.read_parquet(W + "valid_scores_stage2.parquet"))
for c in ("US", "India"):
    x = v.filter(pl.col("country") == c)
    print(c, x.group_by("k", "dtV").agg(pl.len().alias("n"), pl.col("label").mean().round(4).alias("prec"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")).sort("k", "dtV").rows())
    y = x.filter((pl.col("k") == "content") & (pl.col("label") == 1))
    print("  true content swaps sample", y.select("n1", "nq").head(12).rows())
t = prep("test", pl.read_parquet(W + "test_scores_stage2_v9.parquet"))
x = t.filter(pl.col("country") == "France").with_columns(pl.col("at").is_in(GFR).alias("G"))
print("FR v9", x.group_by("k", "G").agg(pl.len().alias("n"), (pl.col("p2") >= 0.75).sum().alias("acc_n"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")).sort("k", "G").rows())
print("FR content at vocab size", x.filter(~pl.col("G") & (pl.col("k") == "content"))["at"].n_unique())
print("FR dt in G (swap generic->other)", x.filter(pl.col("dt").is_in(GFR)).group_by("dt").agg(pl.len(), (pl.col("p2") >= 0.75).mean().round(3)).sort("len", descending=True).head(10).rows())
