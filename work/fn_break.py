import sys
sys.path.insert(0, ".")
import polars as pl
from src.model import load_model, predict
from src.config import is_valid_expr
W = "../work/"
model, feats, T = load_model("lgb_v5cf")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats)))
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = pl.read_parquet(W + "gt_pairs.parquet").join(s1, on="s1_id")
top = va.sort("p", descending=True).unique("s23_id", keep="first").select("s23_id", pl.col("s1_id").alias("top_s1"), pl.col("p").alias("top_p"))
inc = va.select("s1_id", "s23_id", pl.col("p").alias("true_p"))
d = gt.join(inc, on=["s1_id", "s23_id"], how="left").join(top, on="s23_id", how="left")
d = d.with_columns(pl.when(pl.col("true_p").is_null()).then(pl.lit("not_in_cand"))
    .when(pl.col("top_s1") != pl.col("s1_id")).then(pl.when(pl.col("top_p") >= T).then(pl.lit("wrong_s1_assigned")).otherwise(pl.lit("wrong_top_unassigned")))
    .when(pl.col("top_p") < T).then(pl.lit("top_below_t")).otherwise(pl.lit("TP")).alias("kind"))
print(d.group_by("country", "kind").len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("share")).sort("country", "kind"))
# not-in-cand: were they in the unpruned blocking candidates? inspect some
cols = ["entity_id", "business_name", "business_address"]
s1n = pl.read_parquet(W + "norm_train_s1.parquet", columns=cols).rename({"entity_id": "s1_id", "business_name": "n1", "business_address": "a1"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=cols) for k in (2, 3)]).rename({"entity_id": "s23_id"})
x = d.filter(pl.col("kind") == "not_in_cand").sample(25, seed=4).join(s1n, on="s1_id").join(q, on="s23_id")
for r in x.iter_rows(named=True):
    print(f"S1: {r['n1']} | {r['a1']}\n Q: {r['business_name']} | {r['business_address']}")
x = d.filter(pl.col("kind") == "top_below_t").sample(15, seed=4).join(s1n, on="s1_id").join(q, on="s23_id")
print("---- top below t")
for r in x.iter_rows(named=True):
    print(f"p={r['top_p']:.3f} S1: {r['n1']} | {r['a1']}\n Q: {r['business_name']} | {r['business_address']}")
