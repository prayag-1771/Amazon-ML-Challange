import sys
sys.path.insert(0, ".")
import numpy as np
import polars as pl
from src.model import load_model, predict, assign
from src.metric import macro_f05
from src.io_utils import load_ground_truth
from src.config import is_valid_expr

W = "../work/"
model, feats, T = load_model("lgb_v3")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats)))
gt = load_ground_truth()
s1v = pl.read_parquet(W + "raw_train_s1.parquet").filter(is_valid_expr("entity_id")).select(pl.col("entity_id").alias("s1_id"), "country")
print("baseline", macro_f05(s1v["s1_id"], assign(va, T), gt, by=s1v))

def ef_select(df, m=0.0, beta_w=0.25, pmin=0.0):
    """Per S1: choose k maximising 1.25*sum_topk(p)/(beta_w*(sum_all p + m) + k), k=0 -> P(no true)."""
    top = df.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= pmin)
    top = top.sort(["s1_id", "p"], descending=[False, True]).with_columns(
        pl.col("p").cum_sum().over("s1_id").alias("cs"),
        pl.col("p").cum_count().over("s1_id").alias("k"),
        pl.col("p").sum().over("s1_id").alias("sp"),
        (1 - pl.col("p")).log().sum().over("s1_id").alias("lq"),
    )
    top = top.with_columns((1.25 * pl.col("cs") / (beta_w * (pl.col("sp") + m) + pl.col("k"))).alias("ef"),
                           (pl.col("lq").exp() * np.exp(-m)).alias("ef0"))
    best = top.group_by("s1_id").agg(pl.col("ef").max().alias("efmax"), pl.col("ef0").first())
    kbest = top.join(best, on="s1_id").filter((pl.col("ef") == pl.col("efmax")) & (pl.col("efmax") > pl.col("ef0"))) \
        .group_by("s1_id").agg(pl.col("k").min().alias("kb"))
    sel = top.join(kbest, on="s1_id").filter(pl.col("k") <= pl.col("kb"))
    return sel.select("s1_id", "s23_id", "p")

for m in (0.0, 0.05, 0.1, 0.2):
    for pmin in (0.0, 0.05):
        r = macro_f05(s1v["s1_id"], ef_select(va, m=m, pmin=pmin), gt, by=s1v)
        print(f"m={m} pmin={pmin} f05={r['f05']:.5f} P={r['precision_micro']:.4f} R={r['recall_micro']:.4f} "
              f"single={r['singleton_acc']:.4f} US={r.get('f05_US'):.5f} IN={r.get('f05_India'):.5f}", flush=True)
