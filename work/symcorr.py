"""Label-shift correction from house-number shift symmetry (true matches shift up/down equally; distractors shift up)."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl
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
    top = top.with_columns(pl.when(ad == 1).then(pl.lit("d1")).when(ad == 2).then(pl.lit("d2")).when(ad.is_in(D3)).then(pl.lit("d3")).when(ad.is_between(1, 25)).then(pl.lit("dx")).otherwise(None).alias("dg"),
                           (pl.col("dif") > 0).alias("up"))
    return top, c
def counts(top, lo=0.02):
    x = top.filter(pl.col("dg").is_not_null() & (pl.col("p2") > lo))
    g = x.group_by(K + ["up"]).len()
    return g.filter(pl.col("up")).drop("up").rename({"len": "nu"}).join(g.filter(~pl.col("up")).drop("up").rename({"len": "nd"}), on=K, how="full", coalesce=True).fill_null(0)
def prior(cnt, R):
    # R: (country, dg) -> true_up / n_down
    return cnt.join(R, on=["country", "dg"], how="left").with_columns(
        (pl.col("R") * pl.col("nd") / pl.col("nu")).clip(0.01, 0.98).alias("pi"))
v, s1v = prep("valid_scores_stage2.parquet", "train", is_valid_expr("s1_id"))
gt = load_ground_truth()
vv = v.filter(pl.col("dg").is_not_null() & (pl.col("p2") > 0.02))
R = vv.group_by("country", "dg").agg((pl.col("label").filter(pl.col("up")).sum() / (~pl.col("up")).sum()).alias("R"))
print("R", R.sort("country", "dg").rows())
Rf = vv.group_by("dg").agg((pl.col("label").filter(pl.col("up")).sum() / (~pl.col("up")).sum()).alias("R")).with_columns(pl.lit("France").alias("country"))
Rall = pl.concat([R, Rf.select(R.columns)])
cv = prior(counts(v), Rall)
# reference prior per cell from validation (France: pooled US+India cells)
cvp = prior(counts(v.with_columns(pl.lit("France").alias("country"))), Rall)
piv = pl.concat([cv, cvp]).select(K + [pl.col("pi").alias("pi_v"), pl.col("nu").alias("nu_v")])
def apply(top, cnt, strength=1.0, min_nu=20):
    x = top.join(cnt.select(K + ["pi", "nu"]), on=K, how="left").join(piv, on=K, how="left")
    lo = lambda p: np.log(p / (1 - p))
    adj = (pl.col("pi").log() - (1 - pl.col("pi")).log()) - (pl.col("pi_v").log() - (1 - pl.col("pi_v")).log())
    ok = pl.col("up").fill_null(False) & pl.col("pi").is_not_null() & pl.col("pi_v").is_not_null() & (pl.col("nu") >= min_nu) & (pl.col("nu_v") >= min_nu)
    lp = (pl.col("p2").clip(1e-6, 1 - 1e-6).log() - (1 - pl.col("p2").clip(1e-6, 1 - 1e-6)).log())
    x = x.with_columns(pl.when(ok & (adj < 0)).then(1 / (1 + (-(lp + strength * adj)).exp())).otherwise(pl.col("p2")).alias("p3"))
    return x
def f(top, col):
    m = macro_f05(s1v["s1_id"], top.filter(pl.col(col) >= pl.col("country").replace_strict(T)).select("s1_id", "s23_id"), gt, by=s1v)
    return f"{m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f})"
print("val base", f(v, "p2"))
for st in (0.5, 1.0):
    for mn in (20, 50):
        x = apply(v, cv, st, mn); print(f"val self-corr st={st} min={mn}", f(x, "p3"))
t, s1t = prep("test_scores_stage2_v10.parquet", "test")
ct = prior(counts(t), Rall)
for st in (0.5, 1.0):
    x = apply(t, ct, st, 20)
    th = pl.col("country").replace_strict(T)
    print(f"test st={st}", x.group_by("country").agg((pl.col("p2") >= th).sum().alias("acc0"), (pl.col("p3") >= th).sum().alias("acc1")).sort("country").rows())
x = apply(t, ct, 1.0, 20)
x.select("s1_id", "s23_id", "p2", "p3").write_parquet(W + "test_top_symcorr.parquet")
print(ct.join(piv, on=K, how="left").filter(pl.col("nu") > 1000).sort("nu", descending=True).head(30).rows())
