"""Generator vocabulary: tokens added in the query name (xq) or dropped from S1 (x1) on top-1 pairs.
Train US/India with label rate; France test with model p. Raw lowercase name tokens (keep legal forms)."""
import sys; sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
W = "../work/"
gt = pl.read_parquet(W+"gt_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
tok = lambda c: pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}&+]+", " ").str.strip_chars().str.split(" ")
def pairs(split, n=None):
    c = pl.read_parquet(W+f"pruned_{split}.parquet", columns=["q_row","s1_row","p0_qrank","s1_id","s23_id"]).filter(pl.col("p0_qrank")==1)
    if n: c = c.sample(n, seed=1)
    s1 = pl.read_parquet(W+f"norm_{split}_s1.parquet", columns=["country","business_name"])
    q = pl.concat([pl.read_parquet(W+f"norm_{split}_s{k}.parquet", columns=["business_name"]) for k in (2,3)])
    d = pl.concat([c, s1[c["s1_row"].to_numpy()], q[c["q_row"].to_numpy()].rename({"business_name":"nb"})], how="horizontal")
    d = d.with_columns(tok("business_name").alias("ta"), tok("nb").alias("tb"))
    return d.with_columns(pl.col("tb").list.set_difference(pl.col("ta")).alias("xq"), pl.col("ta").list.set_difference(pl.col("tb")).alias("x1"))
tr = pairs("train", 3_000_000).join(gt, on=["s1_id","s23_id"], how="left").with_columns(pl.col("y").fill_null(0))
te = pl.read_parquet(W+"test_scores_lgb_v6k.parquet").rename({"p":"pm"})
tt = pairs("test").join(te, on=["s1_id","s23_id"], how="left").filter(pl.col("country")=="France")
for side in ("xq","x1"):
    for c in ("US","India"):
        e = tr.filter(pl.col("country")==c).select("y", pl.col(side).alias("t")).explode("t").drop_nulls()
        print(side, c, e.group_by("t").agg(pl.len().alias("n"), pl.col("y").mean().alias("ytrue")).sort("n", descending=True).head(60))
    e = tt.select("pm", pl.col(side).alias("t")).explode("t").drop_nulls()
    print(side, "France", e.group_by("t").agg(pl.len().alias("n"), pl.col("pm").mean().alias("pmean"),
          ((pl.col("pm")>0.02)&(pl.col("pm")<0.98)).mean().alias("mid")).sort("n", descending=True).head(80))
