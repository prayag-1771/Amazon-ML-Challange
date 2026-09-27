import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
W = "../work/"
F = ["nc_tset", "nc_ratio", "xq_n", "xq_relmin", "x1_n", "x1_relmin", "state_eq", "num_first_eq", "s1_name_cnt", "q_name_in_s1", "p0", "legal_eq", "src"]
tf = pl.read_parquet(W + "test_feats.parquet", columns=["s1_id", "s23_id"] + F)
sc = pl.read_parquet(W + "test_scores_lgb_v5cf.parquet")
top = sc.sort("p", descending=True).unique("s23_id", keep="first").join(tf, on=["s1_id", "s23_id"])
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "name_full", "addr_core", "country", "business_name", "business_address"]).rename({"entity_id": "s1_id", "name_full": "n1", "addr_core": "a1", "business_name": "N1", "business_address": "A1"})
sib = s1.filter(pl.col("a1") != "").group_by("country", "a1").len("a1_n")
s1 = s1.join(sib, on=["country", "a1"], how="left")
q = pl.concat([pl.read_parquet(W + f"norm_test_s{k}.parquet", columns=["entity_id", "name_full", "addr_core", "business_name", "business_address"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "name_full": "nq", "addr_core": "aq", "business_name": "NQ", "business_address": "AQ"})
d = top.join(s1, on="s1_id").join(q, on="s23_id").filter((pl.col("a1") != "") & (pl.col("a1") == pl.col("aq")) & (pl.col("a1_n") == 1))
d = d.with_columns(pl.col("nq").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).alias("xq"),
                   pl.col("n1").str.split(" ").list.set_difference(pl.col("nq").str.split(" ")).alias("x1"))
d = d.filter((pl.col("xq").list.len() > 0) & (pl.col("x1").list.len() > 0))
d = d.with_columns(pl.col("p").cut([0.1, 0.75, 0.999]).alias("band"))
print(d.group_by("country", "band").agg(pl.len(), *[pl.col(c).mean().round(3) for c in F]).sort("country", "band"))
fr = d.filter(pl.col("country") == "France")
for b in fr["band"].unique().sort().to_list():
    y = fr.filter(pl.col("band") == b)
    print("== France band", b, len(y))
    print("  xq:", y.select(pl.col("xq").explode()).group_by("xq").len().sort("len", descending=True).head(12).rows())
    print("  x1:", y.select(pl.col("x1").explode()).group_by("x1").len().sort("len", descending=True).head(12).rows())
    for r in y.sample(min(6, len(y)), seed=1).iter_rows(named=True):
        print(f"   p={r['p']:.3f} S1: {r['N1']} | {r['A1']}\n            Q: {r['NQ']} | {r['AQ']}")
