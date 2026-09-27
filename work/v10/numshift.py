"""name= & num!= top-1 pairs: house-number difference (b - a) distribution. Val US/India by label vs France test (vfix acc)."""
import sys
sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W = "../work/"
src = open("../work/v10/frcat.py").read()
exec(src[src.index("def cats"):src.index("def load")])
def prep(split, sc):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums", "street"] if False else ["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = cats(sc.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id"))
    x = x.filter((pl.col("nm") == "name=") & (pl.col("num") == "num!="))
    a = pl.col("a1").cast(pl.Int64, strict=False); b = pl.col("aq").cast(pl.Int64, strict=False)
    d = b - a
    return x.with_columns(d.alias("dif"), pl.when(a.is_null() | b.is_null()).then(pl.lit("nonint")).when(d.abs() <= 2).then(d.cast(pl.String))
        .when(d.abs() <= 25).then(pl.when(d > 0).then(pl.lit("+3..25")).otherwise(pl.lit("-3..25"))).otherwise(pl.lit("far")).alias("db"))
v = prep("train", pl.read_parquet(W + "valid_scores_stage2.parquet"))
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id"))
for c in ("US", "India"):
    n = s1v.filter(pl.col("country") == c).height
    x = v.filter(pl.col("country") == c)
    print(c, x.group_by("db").agg((pl.len() / n * 1000).round(1).alias("perK"), pl.col("label").mean().round(3).alias("prec"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")).sort("db").rows())
    print("   true examples", x.filter(pl.col("label") == 1).select("u1", "uq").head(12).rows())
t = prep("test", pl.read_parquet(W + "test_scores_stage2_v11.parquet"))
m = pl.read_csv(W + "probe11_fr_vfix/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).rename({"source1_entity_id": "s1_id"})
m = m.filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(",").alias("s23_id")).explode("s23_id").select("s1_id", "s23_id").with_columns(pl.lit(True).alias("acc"))
t = t.join(m, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("acc").fill_null(False))
nf = (pl.read_parquet(W + "norm_test_s1.parquet", columns=["country"])["country"] == "France").sum()
x = t.filter(pl.col("country") == "France")
print("FR", x.group_by("db").agg((pl.len() / nf * 1000).round(1).alias("perK"), pl.col("acc").mean().round(3).alias("acc"), pl.col("p2").mean().round(3).alias("p2")).sort("db").rows())
print("   FR examples +1..2 rejected", x.filter(pl.col("db").is_in(["1", "2", "-1", "-2"]) & ~pl.col("acc")).select("n1", "u1", "uq", "p2").head(10).rows())
print("   FR examples nonint", x.filter(pl.col("db") == "nonint").select("u1", "uq", "p2", "acc").head(10).rows())
print("   FR examples far rejected", x.filter((pl.col("db") == "far") & ~pl.col("acc")).select("u1", "uq", "p2").head(10).rows())
