import sys; sys.path.insert(0, "../work")
import polars as pl
from addtok import load
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_width_chars(250)
v = load("train", "valid_scores_stage2.parquet")
t = load("test", "test_scores_stage2_v9.parquet")
def one(d, kind):
    if kind == "add":
        f = (pl.col("add").list.len() == 1) & (pl.col("drop").list.len() == 0); tok = pl.col("add").list.first()
    elif kind == "drop":
        f = (pl.col("drop").list.len() == 1) & (pl.col("add").list.len() == 0); tok = pl.col("drop").list.first()
    else:
        f = (pl.col("drop").list.len() == 1) & (pl.col("add").list.len() == 1); tok = pl.col("drop").list.first() + ">" + pl.col("add").list.first()
    return d.filter(f & (pl.col("p2") > 0.01)).with_columns(tok.alias("tok"))
for kind in ("add", "drop", "swap"):
    a = one(v, kind).group_by("country", "tok").agg(pl.len(), pl.col("label").mean().round(2).alias("lab")).filter(pl.col("len") >= 40)
    print(kind, "VAL"); print(a.sort("country", "len", descending=[False, True]).group_by("country", maintain_order=True).head(30))
    b = one(t.filter(pl.col("country") == "France"), kind).group_by("tok").agg(pl.len(), pl.col("p2").mean().round(2).alias("p2"), (pl.col("p2") >= 0.75).mean().round(2).alias("acc")).filter(pl.col("len") >= 30)
    print(kind, "FR"); print(b.sort("len", descending=True).head(40))
