import sys
sys.path.insert(0, ".")
import numpy as np
import polars as pl
from src.metric import macro_f05
from src.config import is_valid_expr
from src.model import load_model, predict
W = "../work/"
gt = pl.read_parquet(W + "gt_pairs.parquet")
model, feats, t0 = load_model("lgb_v6k")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats))).select("s1_id", "s23_id", "p", "label")
va.write_parquet(W + "valid_scores_lgb_v6k.parquet")
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
top = va.sort("p", descending=True).unique("s23_id", keep="first").join(s1v, on="s1_id", how="semi")
dis = top.join(gt, on="s23_id", how="anti")
rng = np.random.default_rng(0)
for w in (1.0, 1.4, 2.0):
    extra, k, rep = [], w - 1, 0
    while k > 1e-9:
        frac = min(k, 1.0)
        extra.append(dis.filter(pl.Series(rng.random(dis.height) < frac)).with_columns((pl.col("s23_id") + f"_dup{rep}").alias("s23_id")))
        k -= frac; rep += 1
    sim = pl.concat([top] + extra) if extra else top
    print("w", w, [(t, round(macro_f05(s1v["s1_id"], sim.filter(pl.col("p") >= t), gt, by=s1v)["f05"], 5)) for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85)], flush=True)
