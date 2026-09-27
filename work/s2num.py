"""Stage-2: structural features (nrel, add/drop counts) + signed first-house-number difference."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl, lightgbm as lgb
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
import src.stage2 as s2
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
d = pl.read_parquet(W + "s2mix_valid.parquet").with_columns(pl.col("qc").fill_null(""))
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "nums"]).rename({"entity_id": "s1_id", "nums": "u1"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{i}.parquet", columns=["entity_id", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "nums": "uq"})
d = d.join(s1, on="s1_id", how="left").join(q, on="s23_id", how="left")
a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False)
b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
d = d.with_columns((b - a).clip(-100, 100).cast(pl.Float32).fill_null(-999).alias("ndiff")).drop("u1", "uq")
# per query: does any candidate have an equal / up-shifted first number
d = d.with_columns((pl.col("nrel_c") == 0).any().over("s23_id").cast(pl.Int8).alias("q_any_eq"),
                   (pl.col("nrel_c") == 1).sum().over("s23_id").alias("q_n_up"))
def ev(d, col, tag):
    top = d.sort(col, descending=True).unique("s23_id", keep="first")
    r = []
    for t in (0.7, 0.75, 0.8):
        m = macro_f05(s1v["s1_id"], top.filter(pl.col(col) >= t).select("s1_id", "s23_id"), gt, by=s1v)
        r.append(f"t={t} {m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f})")
    print(tag, " | ".join(r), flush=True)
ST = ["nrel_c", "add_n", "drop_n"]
for nm, F in (("struct+nd", s2.F + ST + ["ndiff"]), ("struct+nd+q", s2.F + ST + ["ndiff", "q_any_eq", "q_n_up"])):
    oof = np.zeros(len(d), np.float32)
    for k in range(5):
        tr, te = (d["fold"] != k).to_numpy(), (d["fold"] == k).to_numpy()
        m = lgb.train(s2.P, lgb.Dataset(d.filter(tr).select(F).to_numpy(), d.filter(tr)["label"].to_numpy()), s2.R)
        oof[te] = m.predict(d.filter(te).select(F).to_numpy(), num_threads=14)
    ev(d.with_columns(pl.Series("q", oof)), "q", f"CV {nm}")
    tr, te = (d["qc"] == "US").to_numpy(), (d["qc"] == "India").to_numpy()
    m = lgb.train(s2.P, lgb.Dataset(d.filter(tr).select(F).to_numpy(), d.filter(tr)["label"].to_numpy()), s2.R)
    ev(d.filter(te).with_columns(pl.Series("q", m.predict(d.filter(te).select(F).to_numpy(), num_threads=14))), "q", f"US->IN {nm}")
