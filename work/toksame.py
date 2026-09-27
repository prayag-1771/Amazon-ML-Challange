"""Per extra-query-token: share of top-1 pairs at the exact same address, and model p, by country (test) + valid labels."""
import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
W = "../work/"
gt = pl.read_parquet(W + "gt_pairs.parquet")
def top1(split, sc):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "name_full", "addr_core", "country"]).rename({"entity_id": "s1_id", "name_full": "n1", "addr_core": "a1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "name_full", "addr_core"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "name_full": "nq", "addr_core": "aq"})
    t = sc.sort("p", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
    return t.with_columns(((pl.col("a1") != "") & (pl.col("a1") == pl.col("aq"))).alias("same"),
        pl.col("nq").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).alias("xq"))
te = top1("test", pl.read_parquet(W + "test_scores_lgb_v5cf.parquet"))
x = te.explode("xq").filter(pl.col("xq").is_not_null() & (pl.col("xq") != ""))
agg = x.group_by("country", "xq").agg(pl.len().alias("n"), pl.col("same").mean().round(4).alias("same_sh"),
    pl.col("p").mean().round(3).alias("p_mean"), (pl.col("p") >= 0.75).mean().round(3).alias("assigned"),
    pl.col("p").filter(pl.col("same")).mean().round(3).alias("p_same"), (pl.col("p") >= 0.75).filter(pl.col("same")).mean().round(3).alias("asg_same"))
for c in ("France", "US", "India"):
    print("TEST", c); print(agg.filter(pl.col("country") == c).sort("n", descending=True).head(45))
va = pl.read_parquet(W + "valid_scores_lgb_v5cf.parquet")
tv = top1("train", va.select("s1_id", "s23_id", "p", "label"))
xv = tv.explode("xq").filter(pl.col("xq").is_not_null() & (pl.col("xq") != ""))
av = xv.group_by("country", "xq").agg(pl.len().alias("n"), pl.col("same").mean().round(4).alias("same_sh"), pl.col("label").mean().round(3).alias("true"),
    pl.col("label").filter(pl.col("same")).mean().round(3).alias("true_same"), (pl.col("p") >= 0.75).filter(pl.col("same")).mean().round(3).alias("asg_same"))
for c in ("US", "India"):
    print("VALID", c); print(av.filter(pl.col("country") == c).sort("n", descending=True).head(40))
