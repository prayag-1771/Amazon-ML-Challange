"""Expected macro F0.5 on test under the assumption that stage-2 p2 is calibrated (checked on validation OOF)."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl
from src.config import is_valid_expr
W = "../work/"
v = pl.read_parquet(W + "valid_scores_stage2.parquet")
print(v.with_columns(pl.col("p2").cut([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]).alias("b")).group_by("b").agg(pl.len(), pl.col("p2").mean(), pl.col("label").mean()).sort("b"))
def mc(d, s1, T, n=5, rng=np.random.default_rng(0)):
    """d: s1_id, s23_id, p2 (all candidate pairs). Samples truth per query: at most one S1 (categorical on p2, renormalised if sum>1)."""
    d = d.with_columns(pl.col("p2").sum().over("s23_id").alias("ps"))
    d = d.with_columns(pl.when(pl.col("ps") > 1).then(pl.col("p2") / pl.col("ps")).otherwise(pl.col("p2")).alias("q"))
    d = d.sort("s23_id", "q", descending=[False, True])
    pred = d.unique("s23_id", keep="first").filter(pl.col("p2") >= T).select("s1_id", "s23_id").with_columns(pl.lit(1).alias("pr"))
    qid = d["s23_id"].to_numpy(); q = d["q"].to_numpy()
    starts = np.r_[0, np.flatnonzero(qid[1:] != qid[:-1]) + 1]
    cs = np.cumsum(q); base = np.r_[0, cs][starts]
    res = []
    for _ in range(n):
        u = rng.random(len(starts))
        grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(q)]))
        c = cs - base[grp]
        # chosen index: first row in group where cumulative >= u
        hit = c >= u[grp]
        first = np.r_[True, grp[1:] != grp[:-1]]
        prevhit = np.r_[False, hit[:-1]] & ~first
        # propagate "already hit" within group
        cum = np.maximum.accumulate(np.where(hit, np.arange(len(q)), -1))
        chosen = hit & ~np.r_[False, (cum[:-1] >= starts[grp[1:]]) ] if False else None
        idx = np.flatnonzero(hit)
        gi = grp[idx]; keep = np.r_[True, gi[1:] != gi[:-1]]
        tr = d[idx[keep]].select("s1_id", "s23_id").with_columns(pl.lit(1).alias("tr"))
        j = pred.join(tr, on=["s1_id", "s23_id"], how="full", coalesce=True).fill_null(0)
        g = j.group_by("s1_id").agg(pl.col("pr").sum().alias("np"), pl.col("tr").sum().alias("nt"), (pl.col("pr") * pl.col("tr")).sum().alias("tp"))
        g = s1.join(g, on="s1_id", how="left").fill_null(0)
        P = pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(pl.lit(1.0))
        R = pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(pl.lit(1.0))
        f = g.with_columns(P.alias("P"), R.alias("R")).with_columns(
            pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0).when(pl.col("tp") == 0).then(0.0)
            .otherwise(1.25 * pl.col("P") * pl.col("R") / (0.25 * pl.col("P") + pl.col("R"))).alias("f"))
        res.append(f["f"].mean())
    return np.mean(res)
sv = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
st = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
t = pl.read_parquet(W + "test_scores_stage2_v9.parquet")
for c, T in [("US", 0.75), ("India", 0.80), ("France", 0.75)]:
    s = sv.filter(pl.col("country") == c).select("s1_id")
    if c != "France":
        print(c, "valid MC", round(mc(v.join(s, on="s1_id"), s, T), 5))
    s = st.filter(pl.col("country") == c).select("s1_id")
    print(c, "test MC", round(mc(t.join(s, on="s1_id"), s, T), 5), flush=True)
