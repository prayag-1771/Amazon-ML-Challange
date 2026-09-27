"""Pattern '<core> <LEGAL> <word>' (legal form moved before a trailing word): frequency and truth by country."""
import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config(tbl_rows=40, tbl_width_chars=200)
from src.config import is_valid_expr
W = "../work/"
LEG = r"(?i)\b(llc|inc|ltd|pvt ltd|private limited|limited|corp|co|sas|sarl|eurl|sa|sasu|sci|snc|s\.a\.s\.?|s\.a\.r\.l\.?|e\.u\.r\.l\.?)\s+(et\s+)?([a-zà-ÿ]+)\s*$"
def top1(split, sc):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "business_name"]) for i in (2, 3)]).rename({"entity_id": "s23_id"})
    d = pl.read_parquet(sc)
    t = d.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
    return t.with_columns(pl.col("business_name").str.extract(LEG, 3).str.to_lowercase().alias("suf"))
t = top1("test", W + "test_scores_stage2_v9.parquet").filter(pl.col("country") == "France")
x = t.filter(pl.col("suf").is_not_null())
print("FRANCE top1 with legal-mid pattern", x.height, "of", t.height, "accept", (x["p2"] >= 0.75).mean())
print(x.group_by("suf").agg(pl.len(), (pl.col("p2") >= 0.75).mean().round(3).alias("acc"), pl.col("p2").median().round(3).alias("med")).sort("len", descending=True).head(15))
v = top1("train", W + "valid_scores_stage2.parquet")
x = v.filter(pl.col("suf").is_not_null())
print("VAL legal-mid", x.group_by("country").agg(pl.len(), pl.col("label").mean().round(3).alias("true"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")))
print(x.group_by("country", "suf").agg(pl.len(), pl.col("label").mean().round(3).alias("true"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc")).sort("len", descending=True).head(20))
print(x.sample(12, seed=1).select("country", "business_name", "label", "p2").rows())
