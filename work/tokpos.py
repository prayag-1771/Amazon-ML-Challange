"""France ambiguous tokens: position in the query name (prefix/suffix/middle), added vs swapped, vs same-address share."""
import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(70)
W = "../work/"
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "name_full", "addr_core", "country", "business_name", "business_address"]).rename({"entity_id": "s1_id", "name_full": "n1", "addr_core": "a1", "business_name": "N1", "business_address": "A1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{k}.parquet", columns=["entity_id", "name_full", "addr_core", "business_name", "business_address"]).with_columns(pl.lit(k).alias("src")) for k in (2, 3)]).rename({"entity_id": "s23_id", "name_full": "nq", "addr_core": "aq", "business_name": "NQ", "business_address": "AQ"})
sc = pl.read_parquet(W + "test_scores_lgb_v5cf.parquet")
t = sc.sort("p", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
t = t.with_columns(((pl.col("a1") != "") & (pl.col("a1") == pl.col("aq"))).alias("same"),
    pl.col("nq").str.split(" ").list.set_difference(pl.col("n1").str.split(" ")).alias("xq"),
    pl.col("n1").str.split(" ").list.set_difference(pl.col("nq").str.split(" ")).list.len().alias("nx1"))
toks = {"France": ["groupe", "developpement", "club", "sportive", "sas", "france", "holding", "participations", "fils", "et"],
        "US": ["group", "holdings", "center", "services", "inc"], "India": ["group", "enterprises", "center", "services", "sri"]}
for c, tl in toks.items():
    d = t.filter(pl.col("country") == c)
    for tok in tl:
        x = d.filter(pl.col("xq").list.contains(tok)).with_columns(
            pl.col("nq").str.split(" ").alias("tq"))
        x = x.with_columns(pl.when(pl.col("tq").list.first() == tok).then(pl.lit("prefix")).when(pl.col("tq").list.last() == tok).then(pl.lit("suffix")).otherwise(pl.lit("mid")).alias("pos"),
                           pl.when(pl.col("nx1") == 0).then(pl.lit("added")).otherwise(pl.lit("swap")).alias("kind"))
        g = x.group_by("pos", "kind").agg(pl.len(), pl.col("same").mean().round(3).alias("same"), (pl.col("p") >= 0.75).mean().round(3).alias("asg")).sort("len", descending=True)
        print(c, tok, x.height, g.rows())
fr = t.filter((pl.col("country") == "France") & pl.col("xq").list.contains("groupe"))
for s in (True, False):
    print("== groupe same", s)
    for r in fr.filter(pl.col("same") == s).sample(8, seed=3).iter_rows(named=True):
        print(f"  p={r['p']:.3f} S1: {r['N1']} | {r['A1']}\n           Q: {r['NQ']} | {r['AQ']}")
fr = t.filter((pl.col("country") == "France") & pl.col("xq").list.contains("developpement"))
for s in (True, False):
    print("== developpement same", s)
    for r in fr.filter(pl.col("same") == s).sample(6, seed=3).iter_rows(named=True):
        print(f"  p={r['p']:.3f} S1: {r['N1']} | {r['A1']}\n           Q: {r['NQ']} | {r['AQ']}")
