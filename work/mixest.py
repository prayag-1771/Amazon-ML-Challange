"""Mixture estimate of true share among nrel==eq single-added-token pairs, from label-free cell counts.
Pure-distractor reference: tokens whose eq share is minimal (per country). Checked against train labels."""
import polars as pl
pl.Config.set_tbl_rows(120); pl.Config.set_tbl_width_chars(250)
W = "../work/"
tr = pl.read_parquet(W + "trtok_one.parquet")
import importlib.util, sys
sys.argv = ["x"]
# reuse rel()
src = open(W + "numrel2.py", encoding="utf-8").read().split("one = pl.read_parquet")[0]
exec(src)
tr = rel(tr, "train")
def est(x, lab=True):
    g = x.group_by("tok").agg(pl.len().alias("n"), (pl.col("nrel") == "eq").sum().alias("eq"), (pl.col("nrel") == "up25").sum().alias("up"),
                              *( [pl.col("label").filter(pl.col("nrel") == "eq").mean().round(3).alias("lab_eq"), pl.col("label").mean().round(3).alias("lab")] if lab else []),
                              pl.col("p0" if lab else "p2").filter(pl.col("nrel") == "eq").mean().round(3).alias("p_eq"))
    g = g.filter(pl.col("n") >= 300)
    # distractor reference: tokens with up share > 0.4 -> eq/up ratio, take the lower quartile
    ref = g.filter(pl.col("up") / pl.col("n") > 0.4).with_columns((pl.col("eq") / pl.col("up")).alias("r"))
    r = ref["r"].quantile(0.25)
    return g.with_columns((1 - r * pl.col("up") / pl.col("eq")).clip(0, 1).round(3).alias("est_eq")).sort("n", descending=True), r
for c in ("US", "India"):
    g, r = est(tr.filter(pl.col("country") == c))
    print(c, "ref eq/up", round(r, 4)); print(g.head(40))
te = pl.read_parquet(W + "tetok_one.parquet")
for c in ("US", "India", "France"):
    g, r = est(te.filter(pl.col("country") == c), lab=False)
    print("TEST", c, "ref eq/up", round(r, 4)); print(g.head(30))
