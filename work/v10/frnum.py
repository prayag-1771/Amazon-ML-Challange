"""num~ top-1 pairs: first-number relation (down / up / digit-typo) x name category; France accept vs US/India truth."""
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
W = "../work/"
def rel(x):
    f1 = pl.col("u1").fill_null("").str.split(" ").list.first(); fq = pl.col("uq").fill_null("").str.split(" ").list.first()
    i1 = f1.str.extract(r"^(\d+)").cast(pl.Int64, strict=False); iq = fq.str.extract(r"^(\d+)").cast(pl.Int64, strict=False)
    d = iq - i1
    same_digits = f1.str.split("").list.sort() == fq.str.split("").list.sort()
    return x.with_columns(d.alias("d"), pl.when(same_digits).then(pl.lit("transp")).when(d.is_null()).then(pl.lit("alpha"))
        .when(d < 0).then(pl.when(d >= -25).then(pl.lit("down<=25")).otherwise(pl.lit("down>25")))
        .otherwise(pl.when(d <= 25).then(pl.lit("up<=25")).otherwise(pl.lit("up>25"))).alias("rel"))
t = rel(pl.read_parquet(W+"v10/fr_cat_v12.parquet").filter(pl.col("nu") == "num~"))
v = rel(pl.read_parquet(W+"v10/ui_cat_v12.parquet").filter(pl.col("nu") == "num~"))
K = ["nm2", "rel"]
nmc = pl.when(pl.col("nm") == "name=").then(pl.lit("name=")).when(pl.col("nm") == "+1-1").then(pl.lit("+1-1")).otherwise(pl.lit("other")).alias("nm2")
t = t.with_columns(nmc); v = v.with_columns(nmc)
a = v.group_by(K).agg(pl.len().alias("ui_n"), pl.col("label").mean().alias("ui_true"), pl.col("acc").mean().alias("ui_acc"))
b = t.group_by(K).agg(pl.len().alias("fr_n"), pl.col("acc").mean().alias("fr_acc"), pl.col("p").mean().alias("fr_p"), pl.col("p2").mean().alias("fr_p2"))
print(b.join(a, on=K, how="left").sort(K).with_columns(pl.selectors.float().round(3)))
t.write_parquet(W+"v10/fr_numdiff.parquet")
