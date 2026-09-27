"""Adversarial validation: which features separate test assigned pairs from validation assigned pairs (per country)?"""
import sys
sys.path.insert(0, ".")
import numpy as np, polars as pl, lightgbm as lgb
W = "../work/"
va = pl.read_parquet(W + "valid_feats.parquet").join(pl.read_parquet(W + "valid_scores_lgb_v6k.parquet").select("s1_id","s23_id","p"), on=["s1_id","s23_id"])
te = pl.read_parquet(W + "test_feats.parquet").join(pl.read_parquet(W + "test_scores_lgb_v6k.parquet"), on=["s1_id","s23_id"])
c1 = pl.concat([pl.read_parquet(W + f"norm_{s}_s1.parquet", columns=["entity_id","country"]) for s in ("train","test")]).rename({"entity_id":"s1_id"})
feats = [c for c in te.columns if c not in ("q_row","s1_row","s1_id","s23_id","p","label")]
def top(d): return d.sort("p", descending=True).unique("s23_id", keep="first").join(c1, on="s1_id")
va, te = top(va), top(te)
for lo, hi in ((0.7, 1.01), (0.02, 0.7)):
  for c in ("US","India"):
    a = va.filter((pl.col("country")==c)&(pl.col("p")>=lo)&(pl.col("p")<hi)); b = te.filter((pl.col("country")==c)&(pl.col("p")>=lo)&(pl.col("p")<hi))
    b = b.sample(min(len(b), len(a)*3), seed=0)
    X = pl.concat([a.select(feats), b.select(feats)]).to_numpy().astype(np.float32); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    idx = np.random.default_rng(0).permutation(len(y)); cut = int(.7*len(y))
    m = lgb.train(dict(objective="binary", verbose=-1, num_leaves=31, learning_rate=0.1), lgb.Dataset(X[idx[:cut]], y[idx[:cut]]), 100)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y[idx[cut:]], m.predict(X[idx[cut:]]))
    imp = sorted(zip(feats, m.feature_importance("gain")), key=lambda x: -x[1])[:8]
    print(c, lo, hi, "n_val", len(a), "auc", round(auc,4), [(f, int(g)) for f, g in imp], flush=True)
    for f, _ in imp[:4]:
        print("   ", f, "val", a[f].describe().filter(pl.col("statistic").is_in(["mean","25%","50%","75%"]))["value"].round(3).to_list(), "test", b[f].describe().filter(pl.col("statistic").is_in(["mean","25%","50%","75%"]))["value"].round(3).to_list())
