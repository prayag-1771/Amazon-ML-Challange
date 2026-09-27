import sys; sys.path.insert(0, ".")
import polars as pl
from src.stage2 import struct_feats
W = "../work/"
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
def load(split, sc):
    d = pl.read_parquet(sc)
    top = d.sort("p2", descending=True).unique("s23_id", keep="first")
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core"]).rename({"entity_id": "s1_id", "name_core": "n1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq"})
    top = struct_feats(top, split).join(s1, on="s1_id").join(q, on="s23_id")
    t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
    return top.with_columns(tq.list.set_difference(t1).list.first().alias("a"), t1.list.set_difference(tq).list.first().alias("d"))
mt = {sp: pl.read_parquet(W + f"mixtab_{sp}.parquet").filter(pl.col("side") == "add").select("country", pl.col("tok").alias("a"), "est") for sp in ("train", "test")}
v = load("train", W + "valid_scores_stage2.parquet").join(mt["train"], on=["country", "a"], how="left")
t = load("test", W + "test_scores_stage2_v10.parquet").join(mt["test"], on=["country", "a"], how="left")
for nm, cond in (("swap1 eq", (pl.col("add_n") == 1) & (pl.col("drop_n") == 1) & (pl.col("nrel_c") == 0)),
                 ("add1 eq", (pl.col("add_n") == 1) & (pl.col("drop_n") == 0) & (pl.col("nrel_c") == 0)),
                 ("add1 up", (pl.col("add_n") == 1) & (pl.col("drop_n") == 0) & (pl.col("nrel_c") == 1))):
    print("=====", nm)
    print("US val", v.filter(cond & (pl.col("country") == "US")).group_by("a").agg(pl.len(), pl.col("label").mean().round(3).alias("pos"), (pl.col("p2") >= .75).mean().round(3).alias("acc"), pl.col("est").first().round(3)).sort("len", descending=True).head(15))
    print("FR test", t.filter(cond & (pl.col("country") == "France")).group_by("a").agg(pl.len(), (pl.col("p2") >= .75).mean().round(3).alias("acc"), pl.col("est").first().round(3)).sort("len", descending=True).head(15))
