"""Where is the validation F0.5 loss? Oracle variants of the v6 assignment."""
import sys
sys.path.insert(0, ".")
import polars as pl
from src.model import load_model, predict
from src.metric import macro_f05
from src.config import is_valid_expr
W = "../work/"
model, feats, T = load_model("lgb_v5cf")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats)))
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = pl.read_parquet(W + "gt_pairs.parquet").join(s1, on="s1_id", how="semi")
top = va.sort("p", descending=True).unique("s23_id", keep="first")
base = top.filter(pl.col("p") >= T)
def f(pred, name):
    m = macro_f05(s1["s1_id"], pred, gt, by=s1)
    print(f"{name:40s} f05={m['f05']:.5f} US={m.get('f05_US',0):.5f} IN={m.get('f05_India',0):.5f} P={m['precision_micro']:.5f} R={m['recall_micro']:.5f}")
f(base, "v6")
tp = base.join(gt, on=["s1_id", "s23_id"], how="semi")
f(tp, "no false positives")
f(pl.concat([base.select("s1_id", "s23_id"), va.join(gt, on=["s1_id", "s23_id"], how="semi").select("s1_id", "s23_id")]).unique(), "+ all true pairs in candidates")
f(pl.concat([tp.select("s1_id", "s23_id"), va.join(gt, on=["s1_id", "s23_id"], how="semi").select("s1_id", "s23_id")]).unique(), "perfect on candidates")
f(gt, "perfect")
# row-order / id structure check
raw2 = pl.read_parquet(W + "raw_train_s2.parquet", columns=["entity_id"]).with_row_index("row")
g = pl.read_parquet(W + "gt_pairs.parquet").join(raw2.rename({"entity_id": "s23_id"}), on="s23_id")
g = g.sort("s1_id", "row").with_columns(pl.col("row").diff().over("s1_id").alias("dr"))
print("S2 row gap between records of same S1: median", g["dr"].median(), " share gap<=5:", (g["dr"] <= 5).mean())
g = g.with_columns(pl.col("s1_id").str.extract(r"(\d+)").cast(pl.Int64).alias("n1"), pl.col("s23_id").str.extract(r"(\d+)").cast(pl.Int64).alias("n2"))
print("corr(S1 id number, S2 id number):", g.select(pl.corr("n1", "n2")).item(), " corr(S1 id, row):", g.select(pl.corr("n1", "row")).item())
