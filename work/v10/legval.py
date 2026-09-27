import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
exec(open("../work/v10/street.py").read().split("v = st(")[0])
W = "../work/"
vf = pl.read_parquet(W + "valid_feats.parquet", columns=["q_row", "s1_row", "s1_id", "s23_id"])
h = vf.join(pl.read_parquet(W + "ce_hard_s1_valid.parquet"), on=["q_row", "s1_row"]).select("s1_id", "s23_id", "ce_h")
v = st("train", pl.read_parquet(W + "valid_scores_stage2.parquet"), True).filter(is_valid_expr("s1_id")).join(h, on=["s1_id", "s23_id"], how="left")
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "legal"]).rename({"entity_id": "s1_id", "legal": "l1"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{i}.parquet", columns=["entity_id", "legal"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "legal": "lq"})
v = v.join(s1, on="s1_id").join(q, on="s23_id").filter((pl.col("l1").fill_null("") == "") & (pl.col("lq").fill_null("") != "") & (pl.col("ov") > .34))
v = v.with_columns(pl.col("ce_h").cut([-4, -2, 0, 2]).alias("hb"), (pl.col("u1") == pl.col("uq")).alias("numall"))
T = {"US": .75, "India": .8}
for c in ("US", "India"):
    x = v.filter(pl.col("country") == c)
    print(c, x.height)
    print(x.group_by("hb").agg(pl.len(), pl.col("label").mean().round(3).alias("prec"), (pl.col("p2") >= T[c]).mean().round(3).alias("accr"),
          pl.col("label").filter(pl.col("p2") < T[c]).mean().round(3).alias("prec_rej"), (pl.col("p2") < T[c]).sum().alias("nrej")).sort("hb"))
    print(x.filter(pl.col("numall")).group_by((pl.col("p2") >= T[c]).alias("acc")).agg(pl.len(), pl.col("label").mean().round(3)))
    print(x.group_by("lq").agg(pl.len(), pl.col("label").mean().round(3), (pl.col("ce_h") < 0).mean().round(3).alias("ceneg")).sort("len", descending=True).head(8))
f = pl.read_parquet(W + "v10/legfeat.parquet").with_columns(pl.col("ce_h").cut([-4, -2, 0, 2]).alias("hb"))
print("France"); print(f.group_by("hb").agg(pl.len(), pl.col("acc").mean().round(3), pl.col("p2").filter(~pl.col("acc")).median().round(3).alias("rejp2")).sort("hb"))
