import sys
sys.path.insert(0, ".")
import polars as pl
from src.model import load_model, predict
from src.config import is_valid_expr
W = "../work/"
model, feats, T = load_model("lgb_v5cf")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats)))
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = pl.read_parquet(W + "gt_pairs.parquet").join(s1v, on="s1_id", how="semi")
def stats(s1, pred, name):
    n = pred.group_by("s1_id").len("n")
    d = s1.join(n, on="s1_id", how="left").fill_null(0)
    return d.group_by("country").agg(pl.lit(name).alias("what"), pl.len().alias("s1"), pl.col("n").mean().alias("per_s1"),
        (pl.col("n") == 0).mean().alias("empty"), (pl.col("n") == 1).mean().alias("one"), (pl.col("n") >= 6).mean().alias("ge6"))
top = va.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= T)
te = pl.read_parquet(W + "test_scores_lgb_v5cf.parquet").sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= T)
s1t = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
pl.Config.set_tbl_rows(20)
print(pl.concat([stats(s1v, gt, "valid_true"), stats(s1v, top, "valid_pred"), stats(s1t, te, "test_pred")]).sort("country", "what"))
