"""Top-1 pair categories (name relation x address relation): train truth rate vs model assignment, and France test assignment."""
import sys; sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(400); pl.Config.set_tbl_cols(20)
W = "../work/"
def load(split, scores):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_full", "addr_core", "nums"]).rename({"entity_id": "s1_id", "name_full": "n1", "addr_core": "a1", "nums": "m1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "name_full", "addr_core", "nums"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "name_full": "nq", "addr_core": "aq", "nums": "mq"})
    t = pl.read_parquet(W + scores).sort("p", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
    ta = pl.col("n1").str.split(" "); tb = pl.col("nq").str.split(" ")
    t = t.with_columns(tb.list.set_difference(ta).list.len().alias("nxq"), ta.list.set_difference(tb).list.len().alias("nx1"))
    name = (pl.when(pl.col("n1") == pl.col("nq")).then(pl.lit("eq")).when((pl.col("nxq") == 0) & (pl.col("nx1") == 0)).then(pl.lit("reord"))
            .when(pl.col("nx1") == 0).then(pl.lit("added")).when(pl.col("nxq") == 0).then(pl.lit("dropped")).otherwise(pl.lit("swap")))
    f1 = pl.col("m1").str.split(" ").list.first(); fq = pl.col("mq").str.split(" ")
    addr = (pl.when((pl.col("a1") == "") | (pl.col("aq") == "")).then(pl.lit("empty")).when(pl.col("a1") == pl.col("aq")).then(pl.lit("same"))
            .when((pl.col("m1") == "") | (pl.col("mq") == "")).then(pl.lit("nonum")).when(fq.list.contains(f1)).then(pl.lit("numeq")).otherwise(pl.lit("numdiff")))
    return t.with_columns(name.alias("name"), addr.alias("addr"))
va = load("train", "valid_scores_lgb_v6k.parquet")
from src.config import is_valid_expr
va = va.filter(is_valid_expr("s1_id"))
g = va.group_by("name", "addr").agg(pl.len().alias("n_va"), pl.col("label").mean().round(4).alias("truth"), (pl.col("p") >= 0.7).mean().round(4).alias("asg_va"))
te = load("test", "test_scores_lgb_v6k.parquet")
tot = te.group_by("country").len()
h = te.group_by("country", "name", "addr").agg(pl.len(), (pl.col("p") >= 0.7).mean().round(4).alias("asg")).join(tot.rename({"len": "tot"}), on="country").with_columns((pl.col("len") / pl.col("tot")).round(4).alias("share")).drop("tot", "len")
h = h.pivot(on="country", index=["name", "addr"], values=["share", "asg"])
vt = va.height
print(g.with_columns((pl.col("n_va") / vt).round(4).alias("share_va")).join(h, on=["name", "addr"], how="full", coalesce=True).sort("name", "addr"))
