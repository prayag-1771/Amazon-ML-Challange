import polars as pl
W = "../work/"
v = pl.read_parquet(W + "shift_v.parquet"); t = pl.read_parquet(W + "shift_t.parquet")
K = ["country", "dg", "leg", "nsame"]
def cell(x, lab):
    ag = [pl.len().alias("n"), pl.col("acc").sum().alias("a")] + ([pl.col("label").sum().alias("pos"), (pl.col("acc") & (pl.col("label") == 1)).sum().alias("tp")] if lab else [])
    g = x.group_by(K + ["up"]).agg(ag)
    return g.filter(pl.col("up")).drop("up").join(g.filter(~pl.col("up")).drop("up"), on=K, suffix="_dn", how="full", coalesce=True).fill_null(0)
cv, ct = cell(v, True), cell(t, False)
x = ct.join(cv.rename({c: c + "_v" for c in cv.columns if c not in K}), on=K, how="left").fill_null(0)
x = x.with_columns((pl.col("a") - pl.col("n_dn").clip(upper_bound=pl.col("a"))).alias("fp_est"))
for r in x.filter(pl.col("n") > 800).sort("fp_est", descending=True).select(K + ["n", "a", "n_dn", "a_dn", "fp_est", "n_v", "pos_v", "a_v", "tp_v", "n_dn_v", "pos_dn_v"]).head(30).rows(): print(r)
