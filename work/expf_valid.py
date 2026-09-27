import sys, time
sys.path.insert(0, "."); sys.path.insert(0, "../work")
import polars as pl
from src.model import load_model, predict
from src.metric import macro_f05
from src.config import is_valid_expr
from expf import expf_assign
W = "../work/"
model, feats, T = load_model("lgb_v5cf")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats))).select("s1_id", "s23_id", "p")
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = pl.read_parquet(W + "gt_pairs.parquet").join(s1, on="s1_id", how="semi")
def rep(pred, name):
    m = macro_f05(s1["s1_id"], pred, gt, by=s1)
    print(f"{name:34s} f05={m['f05']:.5f} US={m.get('f05_US',0):.5f} IN={m.get('f05_India',0):.5f} P={m['precision_micro']:.5f} R={m['recall_micro']:.5f} single={m['singleton_acc']:.5f}", flush=True)
rep(va.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= T), "v6 t=0.75")
for temp in (1.0,):
    for bias in (0.0, 0.01, 0.03):
        t0 = time.time()
        rep(expf_assign(va, s1["s1_id"], n_samp=200, temp=temp, bias=bias), f"expF temp={temp} bias={bias} ({time.time()-t0:.0f}s)")
