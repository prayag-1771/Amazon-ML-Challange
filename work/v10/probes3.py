"""France category probes on the v9 base (US/India unchanged). Each probe flips ONE category of France top-1 queries.
  sfxacc : accept equal-number queries whose name adds/swaps-in a generic suffix word (groupe, developpement) - model rejects most
  cntacc : accept equal-number non-typo swaps to a content word (club->sportive etc.) - model rejects ~half"""
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
top = top.with_columns(t1.list.set_difference(tq).alias("dl"), tq.list.set_difference(t1).alias("al"))
top = top.with_columns(pl.col("dl").list.first().alias("dt"), pl.col("al").list.first().alias("at"))
top = top.with_columns(pl.struct("dt", "at").map_elements(lambda s: L.normalized_similarity(s["dt"] or "", s["at"] or ""), return_dtype=pl.Float64).alias("sim"))
T = {"US": 0.75, "India": 0.80, "France": 0.75}
top = top.with_columns((pl.col("p2") >= pl.col("country").replace_strict(T, default=0.75)).alias("acc"))
SFX = ["groupe", "developpement"]
FR = (pl.col("country") == "France") & (pl.col("num") == "num=") & (pl.col("p2") >= 0.02) & ~pl.col("acc")
sfx = FR & (((pl.col("nm") == "swap1") & pl.col("at").is_in(SFX)) | ((pl.col("nm") == "addonly") & (pl.col("al").list.len() == 1) & pl.col("at").is_in(SFX)))
cnt = FR & (pl.col("nm") == "swap1") & (pl.col("sim") < 0.5) & ~pl.col("at").is_in(SFX + ["fils", "services", "associes", "france", "cie"])
nfr = (s1["country"] == "France").sum()
for nm, e in [("sfx", sfx), ("cnt", cnt)]:
    x = top.filter(e)
    print(nm, x.height, "S1s", x["s1_id"].n_unique(), "p2 mean", round(x["p2"].mean(), 3), x.group_by("nm").len().rows(), x.group_by("at").len().sort("len", descending=True).head(8).rows())
def write(name, sel):
    m = top.filter(sel)
    g = m.group_by("s1_id").agg(pl.col("s23_id").sort().str.join(",").alias("ids"))
    out = pl.DataFrame({"s1_id": s1["s1_id"]}).join(g, on="s1_id", how="left").with_columns(pl.col("ids").fill_null(""))
    os.makedirs(W + name, exist_ok=True)
    with open(W + name + "/matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")
    print(name, "total pairs", len(m), flush=True)
write("probe_fr_sfxacc", pl.col("acc") | sfx)
write("probe_fr_cntacc", pl.col("acc") | cnt)
