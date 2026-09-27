"""Count-matched per-cell recalibration of up-shifted pairs, from house-number shift symmetry."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl
from scipy.optimize import brentq
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
W = "../work/"
T = {"US": 0.75, "India": 0.80, "France": 0.75}
D3 = [3, 4, 5, 7, 9, 11, 13, 21]
K = ["country", "dg", "leg", "nsame"]
def prep(sc, split, filt=None):
    c = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "legal", "name_core", "nums"]).rename({"entity_id": "s1_id", "legal": "l1", "name_core": "n1", "nums": "u1"})
    if filt is not None: c = c.filter(filt)
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "legal", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "legal": "lq", "name_core": "nq", "nums": "uq"})
    d = pl.read_parquet(W + sc)
    top = d.sort("p2", descending=True).unique("s23_id", keep="first").join(c, on="s1_id").join(q, on="s23_id")
    a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False); b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
    top = top.with_columns((b - a).alias("dif"),
        pl.when(pl.col("l1").fill_null("") == pl.col("lq").fill_null("")).then(pl.lit("same")).when(pl.col("l1").fill_null("") == "").then(pl.lit("add")).when(pl.col("lq").fill_null("") == "").then(pl.lit("drop")).otherwise(pl.lit("chg")).alias("leg"),
        (pl.col("n1") == pl.col("nq")).alias("nsame"))
    ad = pl.col("dif").abs()
    return top.with_columns(pl.when(ad == 1).then(pl.lit("d1")).when(ad == 2).then(pl.lit("d2")).when(ad.is_in(D3)).then(pl.lit("d3")).when(ad.is_between(1, 25)).then(pl.lit("dx")).otherwise(None).alias("dg"),
                            (pl.col("dif") > 0).alias("up")), c
def fix(top, R, slack=1.3, z=3.0, lo=0.02, min_nu=30):
    top = top.with_columns(pl.col("p2").alias("p3"))
    x = top.filter(pl.col("dg").is_not_null() & (pl.col("p2") > lo))
    g = x.group_by(K).agg(pl.col("up").sum().alias("nu"), (~pl.col("up")).sum().alias("nd"), pl.col("p2").filter(pl.col("up")).sum().alias("sp")).join(R, on=["country", "dg"], how="left")
    g = g.with_columns((pl.col("R") * pl.col("nd") * slack + z * (pl.col("R") * pl.col("nd")).sqrt() + 5).alias("tgt"))
    g = g.filter((pl.col("nu") >= min_nu) & (pl.col("sp") > pl.col("tgt")))
    upd = []
    for row in g.iter_rows(named=True):
        m = (pl.col("country") == row["country"]) & (pl.col("dg") == row["dg"]) & (pl.col("leg") == row["leg"]) & (pl.col("nsame") == row["nsame"]) & pl.col("up") & (pl.col("p2") > lo)
        s = x.filter(m)
        lp = np.log(np.clip(s["p2"].to_numpy(), 1e-6, 1 - 1e-6)); lp = lp - np.log1p(-np.exp(lp))
        f = lambda dl: (1 / (1 + np.exp(-(lp + dl)))).sum() - row["tgt"]
        dl = brentq(f, -30, 0)
        upd.append(s.select("s23_id").with_columns(pl.Series("p3n", 1 / (1 + np.exp(-(lp + dl))))))
    if upd:
        u = pl.concat(upd)
        top = top.join(u, on="s23_id", how="left").with_columns(pl.coalesce("p3n", "p3").alias("p3")).drop("p3n")
    return top, g
if __name__ == "__main__":
    gt = load_ground_truth()
    v, s1v = prep("valid_scores_stage2.parquet", "train", is_valid_expr("s1_id"))
    vv = v.filter(pl.col("dg").is_not_null() & (pl.col("p2") > 0.02))
    R = vv.group_by("country", "dg").agg((pl.col("label").filter(pl.col("up")).sum() / (~pl.col("up")).sum()).alias("R"))
    Rp = vv.group_by("dg").agg((pl.col("label").filter(pl.col("up")).sum() / (~pl.col("up")).sum()).alias("R")).with_columns(pl.lit("France").alias("country"))
    R = pl.concat([R, Rp.select(R.columns)])
    R.write_parquet(W + "sym_R.parquet")
    f = lambda top, col: (lambda m: f"{m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f})")(macro_f05(s1v["s1_id"], top.filter(pl.col(col) >= pl.col("country").replace_strict(T)).select("s1_id", "s23_id"), gt, by=s1v))
    print("val base", f(v, "p2"))
    for sl, z in ((1.0, 0.0), (1.0, 2.0), (1.3, 3.0)):
        vf, g = fix(v, R, sl, z); print(f"val fixed slack={sl} z={z}", f(vf, "p3"), "cells", len(g), flush=True)
    t, _ = prep("test_scores_stage2_v10.parquet", "test")
    th = pl.col("country").replace_strict(T)
    for sl, z in ((1.0, 0.0), (1.0, 2.0), (1.3, 3.0)):
        tf, g = fix(t, R, sl, z)
        print(f"test slack={sl} z={z}", tf.group_by("country").agg((pl.col("p2") >= th).sum().alias("acc0"), (pl.col("p3") >= th).sum().alias("acc1"), (pl.col("p2") - pl.col("p3")).sum().round(0).alias("dsum")).sort("country").rows(), "cells", len(g), flush=True)
        if (sl, z) == (1.3, 3.0):
            tf.select("s1_id", "s23_id", "country", "p2", "p3").write_parquet(W + "test_top_symfix.parquet")
            print(g.sort("nu", descending=True).head(25).rows())
