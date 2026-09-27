"""France variant-vocabulary probes on the v9 base (US/India unchanged).
Per-1000-S1 rates of single-token changes at an equal house number show France's variant generator uses
VFR = {fils, groupe, services, associes, developpement, france} (total 144/K; US center/services/service/partners 153/K,
India 140/K). The model rejects most of these (acc .1-.5). Real-word content swaps (club->sportive) at equal number are
distractors on val (US real-word prec .14-.26, India .005), yet France accepts ~40%.
  vacc   : accept top-1 queries whose single added token is in VFR (swap1/addonly, equal number)
  cntrej : reject accepted equal-number swaps to a real frequent word outside VFR (not a typo/abbreviation)
  vfix   : both"""
import sys, os
sys.path.insert(0, ".")
import polars as pl
from rapidfuzz.distance import Levenshtein as L
W = "../work/"
src = open("../work/v10/frcat.py").read()
exec(src[src.index("def cats"):src.index("def load")])
VFR = ["fils", "groupe", "services", "associes", "developpement", "france"]
TAG = "v11"
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
voc = s1.filter(pl.col("country") == "France").select(pl.col("n1").str.split(" ").alias("tok")).explode("tok").group_by("tok").len("fq")
q = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
d = pl.read_parquet(W + f"test_scores_stage2_{TAG}.parquet")
top = cats(d.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id"))
t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
top = top.with_columns(tq.list.set_difference(t1).alias("al"), t1.list.set_difference(tq).list.first().alias("dt"))
top = top.with_columns(pl.col("al").list.first().alias("at"), pl.col("al").list.len().alias("na"))
top = top.with_columns(pl.struct("dt", "at").map_elements(lambda s: L.normalized_similarity(s["dt"] or "", s["at"] or ""), return_dtype=pl.Float64).alias("sim"))
top = top.join(voc.rename({"tok": "at"}), on="at", how="left").with_columns(pl.col("fq").fill_null(0))
T = {"US": 0.75, "India": 0.80, "France": 0.75}
top = top.with_columns((pl.col("p2") >= pl.col("country").replace_strict(T, default=0.75)).alias("acc"))
FR = (pl.col("country") == "France") & (pl.col("num") == "num=") & (pl.col("na") == 1)
vsel = FR & pl.col("nm").is_in(["swap1", "addonly"]) & pl.col("at").is_in(VFR) & (pl.col("p2") >= 0.01) & ~pl.col("acc")
csel = (FR & (pl.col("nm") == "swap1") & ~pl.col("at").is_in(VFR) & (pl.col("sim") < 0.5) & (pl.col("fq") >= 50)
        & (pl.col("at").str.len_chars() >= 4) & ~pl.col("at").str.contains(r"\d") & ~pl.col("dt").str.starts_with(pl.col("at")) & pl.col("acc"))
nfr = (s1["country"] == "France").sum()
for nm, e in [("vacc", vsel), ("cntrej", csel)]:
    x = top.filter(e)
    print(nm, x.height, f"{x.height/nfr*1000:.1f}/K", "p2 mean", round(x["p2"].mean(), 3), x.group_by("nm").len().rows(), x.group_by("at").len().sort("len", descending=True).head(12).rows(), flush=True)
print("examples cntrej", top.filter(csel).select("n1", "nq", "p2").head(10).rows())
print("V at num!= (not touched)", top.filter((pl.col("country") == "France") & (pl.col("num") == "num!=") & (pl.col("na") == 1) & pl.col("at").is_in(VFR) & pl.col("nm").is_in(["swap1"])).group_by("at").agg(pl.len(), pl.col("acc").mean().round(3)).rows())
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
if False:
    write(f"probe_fr_vacc", pl.col("acc") | vsel)
    write(f"probe_fr_cntrej", pl.col("acc") & ~csel)
    write(f"probe_fr_vfix", (pl.col("acc") | vsel) & ~csel)
# v11 base: accepted set = the shipped v11 matching_results (after shift cap), not raw p2 >= T
v11 = pl.read_csv(W + "../output/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).rename({"source1_entity_id": "s1_id"})
v11 = v11.filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(",").alias("s23_id")).explode("s23_id").select("s1_id", "s23_id")
top = top.drop("acc").join(v11.with_columns(pl.lit(True).alias("acc")), on=["s1_id", "s23_id"], how="left").with_columns(pl.col("acc").fill_null(False))
print("v11 matches", v11.height, "reproduced from top-1", int(top["acc"].sum()))
for nm, e in [("vacc", vsel), ("cntrej", csel)]:
    print(nm, "v11", top.filter(e).height)
write("probe11_fr_vacc", pl.col("acc") | vsel)
write("probe11_fr_cntrej", pl.col("acc") & ~csel)
write("probe11_fr_vfix", (pl.col("acc") | vsel) & ~csel)
