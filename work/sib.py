"""Sibling-agreement features: do other candidate queries of the same S1 share this query's deviation?"""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl, lightgbm as lgb
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
import src.stage2 as s2
W = "../work/"

def sib_feats(d, split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = d.select("s1_id", "s23_id").join(s1, on="s1_id", how="left").join(q, on="s23_id", how="left")
    sp = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    x = x.with_columns(sp("n1").alias("t1"), sp("nq").alias("tq"),
                       pl.col("u1").str.split(" ").list.first().alias("f1"), pl.col("uq").str.split(" ").list.first().alias("fq"))
    x = x.with_columns(pl.col("tq").list.set_difference("t1").alias("add"), pl.col("t1").list.set_difference("tq").alias("drop"),
                       pl.len().over("s1_id").alias("nx")).with_row_index("i")
    # token counts among X's candidate queries
    tc = x.select("s1_id", pl.col("tq").alias("tok")).explode("tok").drop_nulls().group_by("s1_id", "tok").len("ct")
    ea = x.select("i", "s1_id", pl.col("add").alias("tok")).explode("tok").drop_nulls().join(tc, on=["s1_id", "tok"], how="left")
    ga = ea.group_by("i").agg((pl.col("ct").min() - 1).alias("sib_add_min"), (pl.col("ct").max() - 1).alias("sib_add_max"))
    ed = x.select("i", "s1_id", "nx", pl.col("drop").alias("tok")).explode("tok").drop_nulls().join(tc, on=["s1_id", "tok"], how="left")
    gd = ed.with_columns((pl.col("nx") - pl.col("ct").fill_null(0) - 1).alias("lack")).group_by("i").agg(pl.col("lack").min().alias("sib_drop_min"), pl.col("lack").max().alias("sib_drop_max"))
    # identical first house number among X's candidate queries
    nc = x.filter(pl.col("fq").is_not_null()).group_by("s1_id", "fq").len("cn")
    x = x.join(nc, on=["s1_id", "fq"], how="left").with_columns(
        pl.when(pl.col("fq").is_null()).then(-1).otherwise(pl.col("cn") - 1).alias("sib_num"),
        pl.when(pl.col("f1").is_null()).then(-1).otherwise(pl.col("f1") == pl.col("fq")).cast(pl.Int8).alias("num_eq1"))
    # exact same name_core as another candidate query of X
    ncq = x.group_by("s1_id", "nq").len("cq")
    x = x.join(ncq, on=["s1_id", "nq"], how="left").with_columns((pl.col("cq") - 1).alias("sib_name"))
    f = x.select("i", "s1_id", "s23_id", "sib_num", "sib_name", "nx").join(ga, on="i", how="left").join(gd, on="i", how="left")
    f = f.with_columns(pl.col("sib_add_min", "sib_add_max", "sib_drop_min", "sib_drop_max").fill_null(-1)).drop("i")
    return d.join(f.rename({"nx": "s1_ncand"}), on=["s1_id", "s23_id"], how="left")

if __name__ == "__main__":
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
    gt = load_ground_truth()
    d = pl.read_parquet(W + "s2mix_valid.parquet").with_columns(pl.col("qc").fill_null(""))
    d = sib_feats(d, "train")
    SB = ["sib_num", "sib_name", "sib_add_min", "sib_add_max", "sib_drop_min", "sib_drop_max"]
    print(d.select(SB).describe())
    def ev(d, col, tag):
        top = d.sort(col, descending=True).unique("s23_id", keep="first")
        r = []
        for t in (0.7, 0.75, 0.8):
            m = macro_f05(s1v["s1_id"], top.filter(pl.col(col) >= t).select("s1_id", "s23_id"), gt, by=s1v)
            r.append(f"t={t} {m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f})")
        print(tag, " | ".join(r), flush=True)
    ST = ["nrel_c", "add_n", "drop_n"]
    for nm, F in (("struct", s2.F), ("struct+sib", s2.F + SB)):
        oof = np.zeros(len(d), np.float32)
        for k in range(5):
            tr, te = (d["fold"] != k).to_numpy(), (d["fold"] == k).to_numpy()
            m = lgb.train(s2.P, lgb.Dataset(d.filter(tr).select(F).to_numpy(), d.filter(tr)["label"].to_numpy()), s2.R)
            oof[te] = m.predict(d.filter(te).select(F).to_numpy(), num_threads=14)
        ev(d.with_columns(pl.Series("q", oof)), "q", f"CV {nm}")
        tr, te = (d["qc"] == "US").to_numpy(), (d["qc"] == "India").to_numpy()
        m = lgb.train(s2.P, lgb.Dataset(d.filter(tr).select(F).to_numpy(), d.filter(tr)["label"].to_numpy()), s2.R)
        ev(d.filter(te).with_columns(pl.Series("q", m.predict(d.filter(te).select(F).to_numpy(), num_threads=14))), "q", f"US->IN {nm}")
        if nm == "struct+sib": print(sorted(zip(F, m.feature_importance("gain").round()), key=lambda x: -x[1])[:15])
