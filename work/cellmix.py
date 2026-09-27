"""Per cell (country, ad, sib, p2 band): negatives among nrel==eq estimated label-free as r * #up25(cell),
r = eq/up ratio of distractors (per country). Compare with the true negative count on validation."""
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
W = "../work/"
R = {"US": 0.0104, "India": 0.4209, "France": 0.0064}
band = pl.col("p2").cut([0.05, 0.2, 0.5, 0.75, 0.9], labels=["a<.05", "b<.2", "c<.5", "d<.75", "e<.9", "f>=.9"]).alias("band")
def cells(d, lab):
    d = d.with_columns(band)
    ag = [(pl.col("nrel") == "eq").sum().alias("eq"), (pl.col("nrel") == "up25").sum().alias("up")]
    if lab: ag.append(((pl.col("nrel") == "eq") & (pl.col("label") == 0)).sum().alias("neg_eq"))
    g = d.group_by("country", "ad", "band").agg(ag)
    return g.with_columns((pl.col("country").replace_strict(R) * pl.col("up")).round(0).alias("est_neg"))
v = cells(pl.read_parquet(W + "top1_val_rel.parquet"), True)
print(v.filter(pl.col("eq") > 300).sort("country", "ad", "band"))
t = cells(pl.read_parquet(W + "top1_test_rel.parquet"), False)
print(t.filter((pl.col("country") == "France") & (pl.col("eq") > 300)).sort("ad", "band"))
