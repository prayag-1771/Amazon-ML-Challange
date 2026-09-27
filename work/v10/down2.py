import sys; sys.path.insert(0, ".")
import polars as pl
W = "../work/"
d = pl.read_parquet(W + "v10/down_fr.parquet").filter((pl.col("nm") == "name~") & (pl.col("nu") == "num="))
sp = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
d = d.with_columns(sp("n1").alias("t1"), sp("nq").alias("tq"))
d = d.with_columns(pl.col("tq").list.set_difference("t1").alias("add"), pl.col("t1").list.set_difference("tq").alias("drp"))
d = d.with_columns(pl.col("add").list.len().alias("na"), pl.col("drp").list.len().alias("nd"))
print(d.group_by("na", "nd").agg(pl.len(), pl.col("p2").median().round(3)).sort("len", descending=True).head(10))
s = d.filter((pl.col("na") == 1) & (pl.col("nd") == 1)).with_columns(pl.col("drp").list.first().alias("o"), pl.col("add").list.first().alias("n"))
print(s.group_by("o", "n").agg(pl.len(), pl.col("p2").median().round(3)).sort("len", descending=True).head(30))
print(s.group_by("n").agg(pl.len()).sort("len", descending=True).head(20).rows())
print(s.group_by("o").agg(pl.len()).sort("len", descending=True).head(20).rows())
d.write_parquet(W + "v10/down2.parquet")
