"""House-number jitter: truth pairs vs same-name non-truth candidates (train) and France test (label-free)."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
W="../work/"
def pairs(split):
    c = pl.read_parquet(W+f"pruned_{split}.parquet", columns=["q_row","s1_row","s1_id","s23_id","p0_qrank"])
    s1 = pl.read_parquet(W+f"norm_{split}_s1.parquet", columns=["country","name_core","nums"]).with_row_index("s1_row")
    q = pl.concat([pl.read_parquet(W+f"norm_{split}_s{k}.parquet", columns=["name_core","nums"]) for k in (2,3)]).with_row_index("q_row").rename({"name_core":"qn","nums":"qnums"})
    d = c.join(s1.with_columns(pl.col("s1_row").cast(pl.UInt32)), on="s1_row").join(q.with_columns(pl.col("q_row").cast(pl.UInt32)), on="q_row")
    d = d.filter((pl.col("name_core")==pl.col("qn")) & (pl.col("nums")!="") & (pl.col("qnums")!=""))
    d = d.with_columns(pl.col("nums").str.split(" ").list.first().cast(pl.Int64, strict=False).alias("a"),
                       pl.col("qnums").str.split(" ").list.first().cast(pl.Int64, strict=False).alias("b")).drop_nulls(["a","b"])
    d = d.with_columns((pl.col("b")-pl.col("a")).alias("diff"), pl.col("a").cut([10,50,200,1000,10000]).alias("mag"))
    return d
tr = pairs("train").join(pl.read_parquet(W+"gt_pairs.parquet").with_columns(pl.lit(1).alias("y")), on=["s1_id","s23_id"], how="left").with_columns(pl.col("y").fill_null(0))
x = tr.filter(pl.col("diff")!=0)
print("train numdiff, same name:", x.group_by("country","y").len().sort("country","y"))
for yv in (1,0):
    z = x.filter(pl.col("y")==yv)
    print("y=",yv, "abs diff quantiles by magnitude")
    print(z.group_by("country","mag").agg(pl.len(), *[pl.col("diff").abs().quantile(qq).alias(f"q{int(qq*100)}") for qq in (.1,.25,.5,.75,.9)],
          (pl.col("diff").abs()<=10).mean().alias("le10")).sort("country","mag"))
print("true diff value counts US", x.filter((pl.col("y")==1)&(pl.col("country")=="US")).select(pl.col("diff").value_counts(sort=True)).unnest("diff").head(30))
te = pairs("test").filter(pl.col("diff")!=0)
sc = pl.read_parquet(W+"test_scores_lgb_v6k.parquet")
te = te.join(sc, on=["s1_id","s23_id"], how="left")
print("France test magnitude dist of S1 number", te.group_by("country","mag").len().sort("country","mag"))
print(te.filter(pl.col("country")=="France").group_by("mag").agg(pl.len(), *[pl.col("diff").abs().quantile(qq).alias(f"q{int(qq*100)}") for qq in (.1,.25,.5,.75,.9)], (pl.col("p")>=0.7).mean().alias("assigned")).sort("mag"))
print(tr.group_by("country","mag").len().sort("country","mag"))
