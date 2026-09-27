"""Per-S1 density of query argmax p by bin: validation vs test (label-shift check)."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
W="../work/"
bins=[0.02,0.1,0.3,0.5,0.7,0.9,0.98,0.999]
def dens(sc, s1, tag):
    b = sc.sort("p", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id")
    n = s1.group_by("country").len("ns1")
    return (b.with_columns(pl.col("p").cut(bins).alias("b")).group_by("country","b").len()
            .join(n, on="country").with_columns((pl.col("len")/pl.col("ns1")).alias(tag)).select("country","b",tag))
v1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country"]).rename({"entity_id":"s1_id"}).filter(is_valid_expr("s1_id"))
t1 = pl.read_parquet(W+"norm_test_s1.parquet", columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
vs = pl.read_parquet(W+"valid_scores_lgb_v6k.parquet")
# restrict valid queries to those whose argmax is a valid S1 (dens joins on s1)
v = dens(vs.drop("label"), v1, "val")
t = dens(pl.read_parquet(W+"test_scores_lgb_v6k.parquet"), t1, "test")
x = v.join(t, on=["country","b"], how="full", coalesce=True).sort("country","b")
print(x.with_columns((pl.col("test")-pl.col("val")).alias("diff")))
# truth rate per bin on val (argmax query)
b = vs.sort("p", descending=True).unique("s23_id", keep="first").join(v1, on="s1_id").with_columns(pl.col("p").cut(bins).alias("b"))
print(b.group_by("country","b").agg(pl.col("label").mean().alias("prec")).sort("country","b"))
