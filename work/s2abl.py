"""Stage-2 with label-free mixture features: 5-fold CV, and US-only -> India transfer (France proxy)."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "../work")
import numpy as np, polars as pl, lightgbm as lgb
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth

import src.stage2 as s2
import mixfeat
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
d = pl.read_parquet(W + "s2mix_valid.parquet").with_columns(pl.col("qc").fill_null(""))
MX = ["nrel_c", "add_n", "drop_n", "mx_add", "mx_addf", "mx_drop", "mx_dropf"]
def ev(d, col, tag):
    top = d.sort(col, descending=True).unique("s23_id", keep="first")
    r = []
    for t in (0.7, 0.75, 0.8):
        m = macro_f05(s1v["s1_id"], top.filter(pl.col(col) >= t).select("s1_id", "s23_id"), gt, by=s1v)
        r.append(f"t={t} {m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f})")
    print(tag, " | ".join(r), flush=True)
for nm, F in (("struct", s2.F + ["nrel_c", "add_n", "drop_n"]), ("mxonly", s2.F + ["mx_add", "mx_addf", "mx_drop", "mx_dropf"])):
    oof = np.zeros(len(d), np.float32)
    for k in range(5):
        tr, te = (d["fold"] != k).to_numpy(), (d["fold"] == k).to_numpy()
        m = lgb.train(s2.P, lgb.Dataset(d.filter(tr).select(F).to_numpy(), d.filter(tr)["label"].to_numpy()), s2.R)
        oof[te] = m.predict(d.filter(te).select(F).to_numpy(), num_threads=14)
    ev(d.with_columns(pl.Series("q", oof)), "q", f"CV {nm:5s}")
    # transfer: train on US queries only, score India queries
    tr, te = (d["qc"] == "US").to_numpy(), (d["qc"] == "India").to_numpy()
    m = lgb.train(s2.P, lgb.Dataset(d.filter(tr).select(F).to_numpy(), d.filter(tr)["label"].to_numpy()), s2.R)
    x = d.filter(te).with_columns(pl.Series("q", m.predict(d.filter(te).select(F).to_numpy(), num_threads=14)))
    ev(x, "q", f"US->IN {nm:5s}")
