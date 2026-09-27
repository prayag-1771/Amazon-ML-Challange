import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.stage2 import struct_feats
W = "../work/"
T = {"US": 0.75, "India": 0.80, "France": 0.75}
def prep(sc, split, filt=None):
    c = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "legal"]).rename({"entity_id": "s1_id", "legal": "l1"})
    if filt is not None: c = c.filter(filt)
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "legal", "addr_empty"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "legal": "lq"})
    d = pl.read_parquet(W + sc)
    top = d.sort("p2", descending=True).unique("s23_id", keep="first").join(c, on="s1_id")
    top = struct_feats(top, split).join(q, on="s23_id")
    top = top.with_columns(pl.when(pl.col("l1").fill_null("") == pl.col("lq").fill_null("")).then(pl.lit("same")).when(pl.col("l1").fill_null("") == "").then(pl.lit("add")).when(pl.col("lq").fill_null("") == "").then(pl.lit("drop")).otherwise(pl.lit("chg")).alias("leg"),
                           pl.col("add_n").clip(0, 2), pl.col("drop_n").clip(0, 2), (pl.col("p2") >= pl.col("country").replace_strict(T)).alias("acc"))
    return top, c.group_by("country").len("ns1")
K = ["nrel_c", "leg", "add_n", "drop_n", "addr_empty"]
v, nv = prep("valid_scores_stage2.parquet", "train", is_valid_expr("s1_id"))
t, nt = prep("test_scores_stage2_v10.parquet", "test")
vg = v.filter(pl.col("p2") > 0.05).group_by(["country"] + K).agg(pl.len().alias("vn"), pl.col("label").sum().alias("vpos"), pl.col("acc").sum().alias("vacc"), (pl.col("acc") & (pl.col("label") == 1)).sum().alias("vtp")).join(nv, on="country")
vg = vg.with_columns((pl.col("vn") / pl.col("ns1")).alias("v_per"), (pl.col("vpos") / pl.col("ns1")).alias("vpos_per")).drop("ns1")
tg = t.filter(pl.col("p2") > 0.05).group_by(["country"] + K).agg(pl.len().alias("tn"), pl.col("acc").sum().alias("tacc")).join(nt, on="country").with_columns((pl.col("tn") / pl.col("ns1")).alias("t_per"))
for src in ("US", "India"):
    x = tg.filter(pl.col("country").is_in([src, "France"] if src == "US" else [src])).join(vg.filter(pl.col("country") == src).drop("country"), on=K, how="left")
    x = x.with_columns((pl.col("vpos_per") * pl.col("ns1")).alias("exp_pos"), (pl.col("t_per") / pl.col("v_per")).round(2).alias("ratio"))
    x = x.with_columns(((pl.col("exp_pos") - pl.col("tacc")).round(0)).alias("pos_minus_acc"), (pl.col("vpos") / pl.col("vn")).round(3).alias("vprec"), (pl.col("tacc") / pl.col("tn")).round(3).alias("tacc_r"))
    print(f"### ref {src}")
    print(x.filter(pl.col("tn") > 3000).sort("tn", descending=True).select("country", *K, "tn", "vn", "ratio", "vprec", "tacc_r", pl.col("exp_pos").round(0), "tacc", "pos_minus_acc").head(45).rows())
