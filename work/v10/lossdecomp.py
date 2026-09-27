"""Where is macro F0.5 lost on validation? Attribute each S1's (1-F) to error types."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W = "../work/"
T = {"US": .75, "India": .8}
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth().join(s1.select("s1_id"), on="s1_id", how="semi")
v = pl.read_parquet(W + "valid_scores_stage2.parquet").join(s1, on="s1_id")
top = v.sort("p2", descending=True).unique("s23_id", keep="first")
pred = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T)).select("s1_id", "s23_id")
cand = v.select("s1_id", "s23_id", "p2")
# classify each true pair
g = gt.join(pred.with_columns(pl.lit(1).alias("hit")), on=["s1_id", "s23_id"], how="left")
g = g.join(cand.rename({"p2": "tp2"}), on=["s1_id", "s23_id"], how="left")
g = g.join(top.select("s23_id", pl.col("s1_id").alias("top1"), pl.col("p2").alias("topp2")), on="s23_id", how="left")
g = g.with_columns(pl.when(pl.col("hit") == 1).then(pl.lit("TP")).when(pl.col("tp2").is_null() & pl.col("top1").is_null()).then(pl.lit("FN_noquerycand"))
    .when(pl.col("tp2").is_null()).then(pl.lit("FN_notincand")).when(pl.col("top1") != pl.col("s1_id")).then(pl.lit("FN_wrongtop1")).otherwise(pl.lit("FN_belowthr")).alias("kind"))
# FPs
gt2 = gt.with_columns(pl.lit(1).alias("lab"))
fp = pred.join(gt2, on=["s1_id", "s23_id"], how="left").filter(pl.col("lab").is_null())
fp = fp.join(gt.select(pl.col("s23_id"), pl.col("s1_id").alias("true_s1")), on="s23_id", how="left").with_columns(
    pl.when(pl.col("true_s1").is_null()).then(pl.lit("FP_querysingleton")).otherwise(pl.lit("FP_wrongS1")).alias("kind"))
# per S1 F and marginal loss attribution: loss if we fixed only that error type
per = s1.join(g.group_by("s1_id").agg(pl.len().alias("nt"), (pl.col("kind") == "TP").sum().alias("tp"),
      *[(pl.col("kind") == k).sum().alias(k) for k in ("FN_noquerycand", "FN_notincand", "FN_wrongtop1", "FN_belowthr")]), on="s1_id", how="left")
per = per.join(fp.group_by("s1_id").agg((pl.col("kind") == "FP_querysingleton").sum().alias("FP_querysingleton"), (pl.col("kind") == "FP_wrongS1").sum().alias("FP_wrongS1")), on="s1_id", how="left").fill_null(0)
def F(tp, fp, fn):
    return pl.when((tp + fn) == 0).then((fp == 0).cast(pl.Float64)).when(tp == 0).then(0.0).otherwise(1.25 * tp / (1.25 * tp + 0.25 * fn + fp))
fn_all = pl.col("FN_noquerycand") + pl.col("FN_notincand") + pl.col("FN_wrongtop1") + pl.col("FN_belowthr")
fp_all = pl.col("FP_querysingleton") + pl.col("FP_wrongS1")
per = per.with_columns(F(pl.col("tp"), fp_all, fn_all).alias("F"))
out = {}
for k in ("FN_noquerycand", "FN_notincand", "FN_wrongtop1", "FN_belowthr"):
    per = per.with_columns(F(pl.col("tp") + pl.col(k), fp_all, fn_all - pl.col(k)).alias("F_" + k))
for k in ("FP_querysingleton", "FP_wrongS1"):
    per = per.with_columns(F(pl.col("tp"), fp_all - pl.col(k), fn_all).alias("F_" + k))
ks = ["FN_noquerycand", "FN_notincand", "FN_wrongtop1", "FN_belowthr", "FP_querysingleton", "FP_wrongS1"]
print(per.group_by("country").agg(pl.col("F").mean().round(5), (1 - pl.col("F").mean()).round(5).alias("loss"),
      *[(pl.col("F_" + k) - pl.col("F")).mean().round(5).alias(k) for k in ks], *[pl.col(k).sum().alias("n_" + k) for k in ks]).transpose(include_header=True))
g.write_parquet(W + "v10/val_truepairs.parquet"); fp.write_parquet(W + "v10/val_fp.parquet"); per.write_parquet(W + "v10/val_per_s1.parquet")
