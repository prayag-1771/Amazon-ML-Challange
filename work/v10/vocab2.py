"""Equal-number single-token swaps where the added token is a real, frequent word of the country's S1 name vocabulary
(not a typo/scramble): precision on val (US/IN), acceptance on France test, by added-token frequency bucket."""
import sys
sys.path.insert(0, ".")
import polars as pl
from rapidfuzz.distance import Levenshtein as L
W = "../work/"
src = open("../work/v10/frcat.py").read()
exec(src[src.index("def cats"):src.index("def load")])
V = ["center", "services", "service", "partners"]
def prep(split, scores):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    voc = s1.select("country", pl.col("n1").str.split(" ").alias("tok")).explode("tok").group_by("country", "tok").len("fq")
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = cats(scores.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id"))
    x = x.filter((pl.col("nm") == "swap1") & (pl.col("num") == "num="))
    t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
    x = x.with_columns(t1.list.set_difference(tq).list.first().alias("dt"), tq.list.set_difference(t1).list.first().alias("at"))
    x = x.with_columns(pl.struct("dt", "at").map_elements(lambda s: L.normalized_similarity(s["dt"] or "", s["at"] or ""), return_dtype=pl.Float64).alias("sim"))
    x = x.filter((pl.col("sim") < 0.5) & ~pl.col("at").is_in(V)).join(voc.rename({"tok": "at"}), on=["country", "at"], how="left").with_columns(pl.col("fq").fill_null(0))
    x = x.join(voc.rename({"tok": "dt", "fq": "fd"}), on=["country", "dt"], how="left").with_columns(pl.col("fd").fill_null(0))
    return x.with_columns(pl.col("fq").cut([0, 5, 50, 500], labels=["0", "1-5", "6-50", "51-500", "500+"]).alias("fb"))
v = prep("train", pl.read_parquet(W + "valid_scores_stage2.parquet"))
for c in ("US", "India"):
    x = v.filter(pl.col("country") == c)
    print(c, x.group_by("fb").agg(pl.len().alias("n"), pl.col("label").mean().round(4).alias("prec"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")).sort("fb").rows())
    y = x.filter(pl.col("fq") > 50)
    print("  frequent-word swaps: true", y.filter(pl.col("label") == 1).select("n1", "nq").head(10).rows())
    print("  frequent-word swaps: false", y.filter(pl.col("label") == 0).select("n1", "nq").head(10).rows())
t = prep("test", pl.read_parquet(W + "test_scores_stage2_v9.parquet"))
x = t.filter(pl.col("country") == "France")
print("FR v9", x.group_by("fb").agg(pl.len().alias("n"), (pl.col("p2") >= 0.75).sum().alias("acc_n"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")).sort("fb").rows())
print("FR fq>500 examples", x.filter(pl.col("fq") > 500).select("n1", "nq", "p2").head(12).rows())
