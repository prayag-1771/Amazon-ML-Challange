"""Exact-address top-1 pairs: true-match rate in train by name-diff kind and S1 address-sibling status; test p distribution."""
import sys
sys.path.insert(0, ".")
import polars as pl
from src.model import load_model, predict
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
W = "../work/"
model, feats, T = load_model("lgb_v5cf")
def prep(split, pairs):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "name_full", "addr_core", "country"]).rename({"entity_id": "s1_id", "name_full": "n1", "addr_core": "a1"})
    sib = s1.filter(pl.col("a1") != "").group_by("country", "a1").len("a1_n")
    s1 = s1.join(sib, on=["country", "a1"], how="left").with_columns(pl.col("a1_n").fill_null(1))
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "name_full", "addr_core"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "name_full": "nq", "addr_core": "aq"})
    d = pairs.join(s1, on="s1_id").join(q, on="s23_id").filter(pl.col("a1") != "")
    d = d.with_columns(
        pl.col("nq").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).list.len().alias("nxq"),
        pl.col("n1").str.split(" ").list.set_difference(pl.col("nq").str.split(" ")).list.len().alias("nx1"),
        (pl.col("a1") == pl.col("aq")).alias("same_addr"), (pl.col("a1_n") > 1).alias("has_sib"))
    return d.with_columns(pl.when(pl.col("nxq") == 0).then(pl.lit("no_extra")).when(pl.col("nx1") == 0).then(pl.lit("added")).otherwise(pl.lit("swap")).alias("kind"))
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats))).sort("p", descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id", "p", "label")
v = prep("train", va)
print("VALID top-1, exact same address")
print(v.filter("same_addr").group_by("country", "has_sib", "kind").agg(pl.len(), pl.col("label").mean().alias("true"), (pl.col("p") >= T).mean().alias("assigned"),
      ((pl.col("p") > 0.1) & (pl.col("p") < 0.75)).mean().alias("unc")).sort("country", "has_sib", "kind"))
print("VALID share of queries whose top-1 has exact same addr", v.group_by("country").agg(pl.col("same_addr").mean()))
te = pl.read_parquet(W + "test_scores_lgb_v5cf.parquet").sort("p", descending=True).unique("s23_id", keep="first")
t = prep("test", te)
print("TEST top-1, exact same address")
print(t.filter("same_addr").group_by("country", "has_sib", "kind").agg(pl.len(), (pl.col("p") >= T).mean().alias("assigned"),
      ((pl.col("p") > 0.1) & (pl.col("p") < 0.75)).mean().alias("unc")).sort("country", "has_sib", "kind"))
print("TEST share same addr", t.group_by("country").agg(pl.col("same_addr").mean(), pl.col("has_sib").mean()))
print("TEST uncertain (0.1-0.75) by same_addr/has_sib", t.filter((pl.col("p") > 0.1) & (pl.col("p") < 0.75)).group_by("country", "same_addr", "has_sib").len().with_columns((pl.col("len")/pl.col("len").sum().over("country")).round(3).alias("share")).sort("country", "same_addr", "has_sib"))
