import polars as pl, lightgbm as lgb, json
from src.io_utils import load_ground_truth
from src.config import is_valid_expr
from src.normalize import normalize_source
from src.model import predict, assign
va = pl.read_parquet("../work/valid_feats.parquet")
meta = json.loads(open("../work/lgb_v1.json").read()); m = lgb.Booster(model_file="../work/lgb_v1.lgb")
va = va.with_columns(pl.Series("p", predict(m, va, meta["features"])))
gt = load_ground_truth()
s1 = normalize_source("train", 1).select(pl.col("entity_id").alias("s1_id"), "country").filter(is_valid_expr("s1_id"))
gtv = gt.join(s1, on="s1_id")
pred = assign(va, meta["threshold"])
cand = va.select("s1_id", "s23_id").with_columns(pl.lit(True).alias("in_cand"))
fn = gtv.join(pred.select("s1_id","s23_id",pl.lit(True).alias("hit")), on=["s1_id","s23_id"], how="left") \
        .filter(pl.col("hit").is_null()).join(cand, on=["s1_id","s23_id"], how="left")
print("FN total", len(fn), "of", len(gtv))
# FN buckets: not in candidates / in cand but under threshold / in cand but another S1 won
top = va.sort("p", descending=True).unique("s23_id", keep="first").select("s23_id", pl.col("s1_id").alias("top_s1"), pl.col("p").alias("top_p"))
fn = fn.join(top, on="s23_id", how="left")
fn = fn.with_columns(pl.when(pl.col("in_cand").is_null()).then(pl.lit("not_in_cand"))
      .when(pl.col("top_s1") != pl.col("s1_id")).then(pl.when(pl.col("top_p") >= meta["threshold"]).then(pl.lit("stolen_assigned")).otherwise(pl.lit("other_top_below_t")))
      .otherwise(pl.lit("below_t")).alias("bucket"))
print(fn.group_by("country","bucket").len().sort("country","len", descending=[False,True]))
fp = pred.join(gt.with_columns(pl.lit(True).alias("ok")), on=["s1_id","s23_id"], how="left").filter(pl.col("ok").is_null())
fp = fp.join(gt.rename({"s1_id":"true_s1"}), on="s23_id", how="left")
print("FP total", len(fp), " of which query has a true S1 elsewhere:", fp["true_s1"].is_not_null().sum())
fn.write_parquet("../work/fn_v1.parquet"); fp.write_parquet("../work/fp_v1.parquet")
