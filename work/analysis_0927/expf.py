"""Per-S1 expected-F0.5 decoding vs fixed per-country thresholds, on validation (stage-2 OOF scores p2).
For each S1: its top-1 queries sorted by p2; accept the top-a maximising plug-in E[F0.5]:
  a = 0 : P(no true match) = prod(1 - p) * exp(-lam)
  a > 0 : 1.25 * sum(p_accepted) / (0.25 * (sum(p_all) + lam) + a)
Rescue / India-channel pairs of v15 are kept as they are. Only pairs with p2 in [LO, HI] may change."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "business_entity_resolution"))
import numpy as np
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05

W = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/"
T = {"US": 0.75, "India": 0.80}
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}).filter(is_valid_expr("s1_id"))
gt = pl.read_parquet(W + "gt_pairs.parquet").join(s1v, on="s1_id", how="semi")
pred = pl.read_parquet(W + "pred15_v.parquet")
extra = pred.filter(pl.col("ch") != "base").select("s1_id", "s23_id")
v = pl.read_parquet(W + "valid_scores_stage2.parquet", columns=["s1_id", "s23_id", "p2"])
top = v.sort("p2", descending=True).unique("s23_id", keep="first").join(s1v, on="s1_id")
top = top.with_columns(pl.col("country").replace_strict(T).alias("th"))
base_sel = pl.concat([top.filter(pl.col("p2") >= pl.col("th")).select("s1_id", "s23_id"), extra])
b = macro_f05(s1v["s1_id"], base_sel, gt, by=s1v)
print(f"thresholds (v15 base+channels): {b['f05']:.5f} US {b['f05_US']:.5f} IN {b['f05_India']:.5f}")


def decode(top, lam, lo, hi):
    t = top.sort(["s1_id", "p2"], descending=[False, True])
    s = t["s1_id"].to_numpy()
    p = t["p2"].to_numpy().astype(np.float64)
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    ends = np.r_[starts[1:], len(s)]
    keep = np.zeros(len(s), bool)
    for a0, a1 in zip(starts, ends):
        pp = p[a0:a1]
        forced = int((pp > hi).sum())           # always accepted
        allowed = int((pp >= lo).sum())          # may be accepted
        tot = pp.sum() + lam
        best_a, best = forced, -1.0
        for a in range(forced, allowed + 1):
            if a == 0:
                ef = np.prod(1 - pp) * np.exp(-lam)
            else:
                ef = 1.25 * pp[:a].sum() / (0.25 * tot + a)
            if ef > best:
                best, best_a = ef, a
        keep[a0:a0 + best_a] = True
    return t.filter(pl.Series(keep)).select("s1_id", "s23_id")


for lam in (0.03, 0.06, 0.12):
    for lo, hi in ((0.3, 0.95), (0.1, 0.99)):
        sel = pl.concat([decode(top, lam, lo, hi), extra])
        m = macro_f05(s1v["s1_id"], sel, gt, by=s1v)
        print(f"expF lam={lam} band=[{lo},{hi}]: {m['f05']:.5f} ({m['f05'] - b['f05']:+.5f}) US {m['f05_US'] - b['f05_US']:+.5f} IN {m['f05_India'] - b['f05_India']:+.5f}  pairs {sel.height - base_sel.height:+d}")
