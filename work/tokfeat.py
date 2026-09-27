"""Stage-2 + label-free per-country token-role features. CV on val + France/test acceptance deltas + shift excess."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "../work")
import numpy as np, polars as pl, lightgbm as lgb
import src.stage2 as s2
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W = "../work/"
def tok_feats(d, split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = d.select("s1_id", "s23_id", "p").join(s1, on="s1_id", how="left").join(q, on="s23_id", how="left")
    sp = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False); b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
    x = x.with_columns(sp("n1").alias("t1"), sp("nq").alias("tq"), (b == a).fill_null(False).alias("eq"), (b - a).abs().is_between(1, 25).fill_null(False).alias("sh"))
    x = x.with_columns(pl.col("tq").list.set_difference("t1").alias("A"), pl.col("t1").list.set_difference("tq").alias("D"))
    # label-free rates over each query's stage-1 top-1, per 1000 S1 of the country
    top = x.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("A").list.len() == 1)
    top = top.with_columns(pl.col("A").list.first().alias("at"), pl.col("D").list.first().alias("dt"), pl.col("D").list.len().alias("nd"))
    ns = s1.join(d.select("s1_id").unique(), on="s1_id").group_by("country").len("ns")
    sw = top.filter((pl.col("nd") == 1) & pl.col("eq"))
    st = s1.join(d.select("s1_id").unique(), on="s1_id").select("country", sp("n1").alias("tok")).explode("tok").drop_nulls().group_by("country", "tok").len("s1f")
    for z in (sw.group_by("country", pl.col("at").alias("tok")).len("sw_in"), sw.group_by("country", pl.col("dt").alias("tok")).len("sw_out"),
              top.filter((pl.col("nd") == 0) & pl.col("sh")).group_by("country", pl.col("at").alias("tok")).len("add_sh")):
        st = st.join(z, on=["country", "tok"], how="full", coalesce=True)
    st = st.join(ns, on="country").with_columns([(pl.col(c).fill_null(0) / pl.col("ns") * 1000).cast(pl.Float32).alias(c) for c in ("s1f", "sw_in", "sw_out", "add_sh")])
    st = st.with_columns((pl.col("sw_in") / (pl.col("sw_out") + 0.2)).alias("io")).drop("ns")
    x = x.select("s1_id", "s23_id", "country", "A", "D").with_row_index("i")
    ga = (x.select("i", "country", pl.col("A").alias("tok")).explode("tok").drop_nulls().join(st, on=["country", "tok"], how="left")
          .group_by("i").agg(pl.col("io").fill_null(0).min().alias("tk_add_io"), pl.col("sw_in").fill_null(0).min().alias("tk_add_in"),
                             pl.col("add_sh").fill_null(0).max().alias("tk_add_sh"), pl.col("s1f").fill_null(0).max().alias("tk_add_s1f")))
    gd = (x.select("i", "country", pl.col("D").alias("tok")).explode("tok").drop_nulls().join(st, on=["country", "tok"], how="left")
          .group_by("i").agg(pl.col("sw_out").fill_null(0).max().alias("tk_drop_out"), pl.col("s1f").fill_null(0).max().alias("tk_drop_s1f")))
    f = x.select("i", "s1_id", "s23_id").join(ga, on="i", how="left").join(gd, on="i", how="left").drop("i")
    return d.join(f, on=["s1_id", "s23_id"], how="left"), st
TK = ["tk_add_io", "tk_add_in", "tk_add_sh", "tk_add_s1f", "tk_drop_out", "tk_drop_s1f"]
if __name__ == "__main__":
    gt = load_ground_truth()
    s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
    s1t = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    dv = s2.ce_hard_feats(s2.sib_feats(s2.struct_feats(s2.feats(s2.base("valid")), "train"), "train"), "valid").join(s1v, on="s1_id")
    dv, stv = tok_feats(dv.drop("country"), "train"); dv = dv.join(s1v, on="s1_id")
    qf = dv.filter(pl.col("q_rank") == 1).select("s23_id", (pl.col("s1_id").hash(3) % 5).alias("fold"))
    dv = dv.join(qf, on="s23_id")
    dt = s2.ce_hard_feats(s2.sib_feats(s2.struct_feats(s2.feats(s2.base("test")), "test"), "test"), "test")
    dt, stt = tok_feats(dt, "test"); dt = dt.join(s1t, on="s1_id")
    stv.write_parquet(W + "tk_stats_valid.parquet"); stt.write_parquet(W + "tk_stats_test.parquet")
    th = pl.col("country").replace_strict(s2.T, default=0.75)
    R = pl.read_parquet(W + "shift_R.parquet"); Rp = pl.read_parquet(W + "shift_Rp.parquet")
    for nm, F in (("v11", s2.F), ("v11+tk", s2.F + TK)):
        oof = np.zeros(len(dv), np.float32)
        for k in range(5):
            tr, te = (dv["fold"] != k).to_numpy(), (dv["fold"] == k).to_numpy()
            m = lgb.train(s2.P, lgb.Dataset(dv.filter(tr).select(F).to_numpy(), dv.filter(tr)["label"].to_numpy()), s2.R)
            oof[te] = m.predict(dv.filter(te).select(F).to_numpy(), num_threads=14)
        top = dv.with_columns(pl.Series("p2", oof)).sort("p2", descending=True).unique("s23_id", keep="first")
        mm = macro_f05(s1v["s1_id"], top.filter(pl.col("p2") >= th).select("s1_id", "s23_id"), gt, by=s1v)
        m = lgb.train(s2.P, lgb.Dataset(dv.select(F).to_numpy(), dv["label"].to_numpy()), s2.R)
        m.save_model(W + f"stage2_tk_{nm}.lgb")
        tt = dt.select("s1_id", "s23_id", "p", "country").with_columns(pl.Series("p2", m.predict(dt.select(F).to_numpy(), num_threads=14)))
        tt.write_parquet(W + f"test_scores_tk_{nm}.parquet")
        tp = tt.sort("p2", descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id", "p2", "country")
        tp = s2.shift_cap(s2.shift_cells(tp, "test"), R, Rp)
        x = tp.filter(pl.col("dg").is_not_null() & (pl.col("p2") >= th)).join(R, on=["country", "dg"], how="left").join(Rp.rename({"R": "Rp"}), on="dg", how="left").with_columns(pl.coalesce("R", "Rp").alias("R"))
        ex = x.group_by("country", "dg").agg(pl.col("up").sum().alias("au"), (~pl.col("up")).sum().alias("ad"), pl.col("R").first()).group_by("country").agg((pl.col("au") - pl.col("R") * pl.col("ad")).sum().round(0)).sort("country").rows()
        acc = tp.group_by("country").agg((pl.col("p2") >= th).sum()).sort("country").rows()
        tp.filter(pl.col("p2") >= th).select("s1_id", "s23_id", "country").write_parquet(W + f"tk_acc_{nm}.parquet")
        print(f"{nm:8s} CV {mm['f05']:.5f} (US {mm['f05_US']:.5f} IN {mm['f05_India']:.5f}) P {mm.get('precision', 0):.5f} R {mm.get('recall', 0):.5f}  test excess {ex} acc {acc}", flush=True)
        if nm == "v11+tk": print(sorted(zip(F, m.feature_importance("gain").round()), key=lambda z: -z[1])[:20])
