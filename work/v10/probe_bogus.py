"""Probe: every France S1 gets one (almost surely wrong) candidate = its lowest-p2 candidate; US/India = v9.
France then scores ~0 everywhere, so LB = w_nonFR * F_nonFR; with the empty probe this pins the France singleton rate."""
import sys, os
sys.path.insert(0, ".")
import polars as pl
W = "../work/"
d = pl.read_parquet(W + "test_scores_stage2_v9.parquet")
c = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
d = d.join(c, on="s1_id")
top = d.sort("p2", descending=True).unique("s23_id", keep="first")
T = {"US": 0.75, "India": 0.80}
m = top.filter((pl.col("country") != "France") & (pl.col("p2") >= pl.col("country").replace_strict(T, default=0.75))).select("s1_id", "s23_id")
b = d.filter(pl.col("country") == "France").sort("p2").unique("s1_id", keep="first")
print("France S1", (c["country"] == "France").sum(), "with a candidate", b.height, "bogus p2 max", b["p2"].max(), "mean", b["p2"].mean())
m = pl.concat([m, b.select("s1_id", "s23_id")])
g = m.group_by("s1_id").agg(pl.col("s23_id").sort().str.join(",").alias("ids"))
out = pl.DataFrame({"s1_id": c["s1_id"]}).join(g, on="s1_id", how="left").with_columns(pl.col("ids").fill_null(""))
os.makedirs(W + "probe_fr_bogus", exist_ok=True)
with open(W + "probe_fr_bogus/matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for a, x in out.iter_rows():
        f.write(f"{a}\t{x}\n")
