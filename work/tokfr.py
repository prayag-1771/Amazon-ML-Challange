import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
W = "../work/"
te = pl.read_parquet(W + "tokstat_test.parquet")
tr = pl.read_parquet(W + "tokstat_train.parquet")
for side in ("xq", "x1"):
    print(side, "prior", te.filter(pl.col("side") == side).group_by("country").agg(pl.col("prior").first()).sort("country").rows())
    fr = te.filter((pl.col("country") == "France") & (pl.col("side") == side)).sort("lfreq", descending=True).head(40)
    print(fr.select("tok", "rate", "rel", (pl.col("lfreq").exp() - 1).round().alias("n")))
    us = tr.filter((pl.col("country") == "US") & (pl.col("side") == side) & pl.col("tok").is_in(["group", "ventures", "west", "public", "services", "center", "formerly", "lp", "partners"]))
    print(us.select("tok", "rate", "rel", (pl.col("lfreq").exp() - 1).round().alias("n")))
