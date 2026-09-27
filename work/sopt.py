"""Per-S1 expected-F0.5-optimal set selection vs global threshold (validation, OOF p2)."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
d = pl.read_parquet(W + "valid_scores_stage2.parquet").join(s1v, on="s1_id")
top = d.sort("p2", descending=True).unique("s23_id", keep="first")
T = {"US": 0.75, "India": 0.80}
base = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T)).select("s1_id", "s23_id")
print("base", macro_f05(s1v["s1_id"], base, gt, by=s1v))
# expected true count per S1 from all candidate pairs (soft), plus unblocked misses
mass = d.group_by("s1_id").agg(pl.col("p2").sum().alias("mass"))
rng = np.random.default_rng(0)
def select(top, a=1.0, b=0.0, S=64, pmin=0.05):
    c = top.filter(pl.col("p2") >= pmin).join(mass, on="s1_id").sort(["s1_id", "p2"], descending=[False, True])
    c = c.with_columns(pl.col("p2").cum_count().over("s1_id").alias("k"))
    p = np.clip(c["p2"].to_numpy() ** a, 0, 1); grp = c["s1_id"].to_physical() if False else c["s1_id"].rank("dense").to_numpy() - 1
    k = c["k"].to_numpy(); G = grp.max() + 1
    # other mass (true pairs not in selectable list): mass - sum(selectable p)
    sel_sum = np.bincount(grp, p, G)
    other = np.maximum(mc := c.group_by("s1_id", maintain_order=True).agg(pl.col("mass").first())["mass"].to_numpy() - sel_sum, 0) + b
    ef = np.zeros(len(p)); ef0 = np.zeros(G)
    for s in range(S):
        y = (rng.random(len(p)) < p).astype(np.float64)
        ntrue = np.bincount(grp, y, G) + rng.poisson(other)
        tp = np.zeros(len(p))
        # cumulative tp within group (rows sorted by group then p desc)
        cs = np.cumsum(y); start = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
        off = np.repeat(cs[start] - y[start], np.diff(np.r_[start, len(p)]))
        tp = cs - off; nt = ntrue[grp]
        f = np.where(tp > 0, 1.25 * tp / (0.25 * nt + k), 0.0)
        ef += f; ef0 += (ntrue == 0)
    ef /= S; ef0 /= S
    c = c.with_columns(pl.Series("ef", ef))
    best = c.group_by("s1_id").agg(pl.col("ef").max(), pl.col("k").sort_by("ef").last().alias("kb"))
    best = best.join(pl.DataFrame({"s1_id": c["s1_id"].unique(maintain_order=True), "ef0": ef0}), on="s1_id")
    best = best.with_columns(pl.when(pl.col("ef") > pl.col("ef0")).then(pl.col("kb")).otherwise(0).alias("kb"))
    return c.join(best.select("s1_id", "kb"), on="s1_id").filter(pl.col("k") <= pl.col("kb")).select("s1_id", "s23_id")
for a in (1.0, 1.5, 2.0):
    for b in (0.0, 0.05):
        r = macro_f05(s1v["s1_id"], select(top, a, b), gt, by=s1v)
        print(a, b, {k: round(v, 5) for k, v in r.items()}, flush=True)
