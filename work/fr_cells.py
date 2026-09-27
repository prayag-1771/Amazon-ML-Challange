"""Compare cells (name relation x addr empty x nrel) between val (with labels) and test per country."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.stage2 import struct_feats
W = "../work/"
def load(split, sc):
    d = pl.read_parquet(sc)
    top = d.sort("p2", descending=True).unique("s23_id", keep="first")
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core"]).rename({"entity_id": "s1_id", "name_core": "n1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "addr_empty"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq"})
    top = struct_feats(top, split).join(s1, on="s1_id").join(q, on="s23_id")
    top = top.with_columns(
        pl.when(pl.col("n1") == pl.col("nq")).then(pl.lit("same")).when((pl.col("add_n") > 0) & (pl.col("drop_n") == 0)).then(pl.lit("add"))
        .when((pl.col("add_n") == 0) & (pl.col("drop_n") > 0)).then(pl.lit("drop")).when((pl.col("add_n") == 1) & (pl.col("drop_n") == 1)).then(pl.lit("swap1"))
        .when((pl.col("add_n") == 0) & (pl.col("drop_n") == 0)).then(pl.lit("reord")).otherwise(pl.lit("other")).alias("nr"),
        pl.col("nrel_c").replace_strict({0: "eq", 1: "up", 2: "dn", 3: "far", 4: "miss"}).alias("nrel"))
    return top
v = load("train", W + "valid_scores_stage2.parquet")
t = load("test", W + "test_scores_stage2_v10.parquet")
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_width_chars(250)
gv = v.group_by("country", "nr", "nrel").agg(pl.len().alias("n_v"), pl.col("label").mean().round(3).alias("pos_v"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc_v"),
                                              ((pl.col("p2") >= 0.75) & (pl.col("label") == 0)).sum().alias("fp_v"), ((pl.col("p2") < 0.75) & (pl.col("label") == 1)).sum().alias("fn_v"))
gt_ = t.group_by("country", "nr", "nrel").agg(pl.len().alias("n_t"), (pl.col("p2") >= 0.75).mean().round(3).alias("acc_t"))
tot = t.group_by("country").len().rename({"len": "N"})
gt_ = gt_.join(tot, on="country").with_columns((pl.col("n_t") / pl.col("N")).round(4).alias("sh_t")).drop("N")
totv = v.group_by("country").len().rename({"len": "N"})
gv = gv.join(totv, on="country").with_columns((pl.col("n_v") / pl.col("N")).round(4).alias("sh_v")).drop("N")
fr = gt_.filter(pl.col("country") == "France").drop("country").rename({"n_t": "n_fr", "acc_t": "acc_fr", "sh_t": "sh_fr"})
for c in ("US", "India"):
    x = gv.filter(pl.col("country") == c).join(gt_.filter(pl.col("country") == c).drop("country"), on=["nr", "nrel"], how="full", coalesce=True).join(fr, on=["nr", "nrel"], how="full", coalesce=True)
    print(c); print(x.sort("n_fr", descending=True, nulls_last=True).drop("country"))
