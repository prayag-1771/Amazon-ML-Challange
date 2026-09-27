"""LB probes: v9 with only the France threshold changed (US/India identical to v9)."""
import sys
sys.path.insert(0, ".")
import polars as pl
W = "../work/"
d = pl.read_parquet(W + "test_scores_stage2_v9.parquet")
c = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
d = d.join(c, on="s1_id")
top = d.sort("p2", descending=True).unique("s23_id", keep="first")
T = {"US": 0.75, "India": 0.80}
s1 = c["s1_id"]
def write(name, tfr):
    m = top.filter(pl.col("p2") >= pl.col("country").replace_strict({**T, "France": tfr}, default=0.75))
    g = m.group_by("s1_id").agg(pl.col("s23_id").sort().str.join(",").alias("ids"))
    out = pl.DataFrame({"s1_id": s1}).join(g, on="s1_id", how="left").with_columns(pl.col("ids").fill_null(""))
    import os; os.makedirs(W + name, exist_ok=True)
    with open(W + name + "/matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")
    fr = m.filter(pl.col("country") == "France")
    print(name, "France pairs", len(fr), "per S1", round(len(fr) / (c["country"] == "France").sum(), 4), flush=True)
#write("probe_v9_check", 0.75)
#write("probe_fr_t95", 0.95)
#write("probe_fr_t40", 0.40)
fr = top.filter(pl.col("country") == "France")
for lo, hi in [(0.4, 0.75), (0.75, 0.95), (0.95, 1.01)]:
    print(lo, hi, fr.filter(pl.col("p2").is_between(lo, hi, closed="left")).height)
for lo, hi in [(0.95, 0.99), (0.99, 0.999), (0.999, 0.9999), (0.9999, 1.01)]:
    print(lo, hi, fr.filter(pl.col("p2").is_between(lo, hi, closed="left")).height)
write("probe_fr_t99", 0.99)
write("probe_fr_t999", 0.999)
