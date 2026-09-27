"""Targeted France LB probes on the v9 base (US/India unchanged): accept / reject the generic-word swap category."""
import sys, os
sys.path.insert(0, ".")
import polars as pl
from rapidfuzz.distance import Levenshtein as L
W = "../work/"
exec(open("../work/v10/frcat.py").read().split("def load")[0].split("W = \"../work/\"")[1])
d = pl.read_parquet(W + "test_scores_stage2_v9.parquet")
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
top = cats(d.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id"))
t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
top = top.with_columns(t1.list.set_difference(tq).list.first().alias("dt"), tq.list.set_difference(t1).list.first().alias("at"))
top = top.with_columns(pl.struct("dt", "at").map_elements(lambda s: L.normalized_similarity(s["dt"] or "", s["at"] or ""), return_dtype=pl.Float64).alias("sim"))
T = {"US": 0.75, "India": 0.80, "France": 0.75}
top = top.with_columns((pl.col("p2") >= pl.col("country").replace_strict(T, default=0.75)).alias("acc"))
gsw = (pl.col("country") == "France") & (pl.col("nm") == "swap1") & (pl.col("num") == "num=") & (pl.col("sim") < 0.5)
nfr = (s1["country"] == "France").sum()
def write(name, sel):
    m = top.filter(sel)
    g = m.group_by("s1_id").agg(pl.col("s23_id").sort().str.join(",").alias("ids"))
    out = pl.DataFrame({"s1_id": s1["s1_id"]}).join(g, on="s1_id", how="left").with_columns(pl.col("ids").fill_null(""))
    os.makedirs(W + name, exist_ok=True)
    with open(W + name + "/matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")
    fr = m.filter(pl.col("country") == "France")
    print(name, "France pairs", len(fr), "per S1", round(len(fr) / nfr, 4), "total", len(m), flush=True)
write("probe_v9_check2", pl.col("acc"))
write("probe_fr_swapacc", pl.col("acc") | (gsw & (pl.col("p2") >= 0.02)))
write("probe_fr_swaprej", pl.col("acc") & ~gsw)
