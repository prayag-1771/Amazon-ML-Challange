"""Per-1000-S1 rates by (name relation, number relation, token kind): val US/India true & accepted vs France test accepted
(vfix). Generator is shared across countries, so France accepted/K should track US/India true/K; gaps flag errors."""
import sys
sys.path.insert(0, ".")
import polars as pl
from rapidfuzz.distance import Levenshtein as L
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W = "../work/"
src = open("../work/v10/frcat.py").read()
exec(src[src.index("def cats"):src.index("def load")])
VAR = ["center", "services", "service", "partners", "fils", "groupe", "associes", "developpement", "france"]
def kind(x):
    ocr = pl.col("at").str.contains(r"\d") | (pl.col("at").str.len_chars() <= 2)
    return x.with_columns(pl.when(~pl.col("nm").is_in(["swap1", "addonly"])).then(pl.lit("-"))
        .when(pl.col("at").is_in(VAR)).then(pl.lit("var")).when(ocr | (pl.col("sim") >= 0.5)).then(pl.lit("typo")).otherwise(pl.lit("word")).alias("k"))
def prep(split, pairs):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = cats(pairs.join(s1, on="s1_id").join(q, on="s23_id"))
    t1 = pl.col("n1").fill_null("").str.split(" "); tq = pl.col("nq").fill_null("").str.split(" ")
    x = x.with_columns(t1.list.set_difference(tq).list.first().alias("dt"), tq.list.set_difference(t1).list.first().alias("at"))
    x = x.with_columns(pl.struct("dt", "at").map_elements(lambda s: L.normalized_similarity(s["dt"] or "", s["at"] or ""), return_dtype=pl.Float64).alias("sim"))
    return kind(x)
K = ["nm", "num", "k"]
# val: top-1 pairs with label + all true pairs
v = pl.read_parquet(W + "valid_scores_stage2.parquet")
top = v.sort("p2", descending=True).unique("s23_id", keep="first")
T = {"US": 0.75, "India": 0.80}
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth(); gt = gt.rename({gt.columns[1]: "s23_id"}) if gt.columns[1] != "s23_id" else gt
gt = gt.join(s1v, on="s1_id").select("s1_id", "s23_id")
tv = prep("train", gt).with_columns(pl.lit(1).alias("true"))
av = prep("train", top.join(s1v, on="s1_id").filter(pl.col("p2") >= pl.col("country").replace_strict(T)).select("s1_id", "s23_id", "label"))
m = pl.read_csv(W + "probe11_fr_vfix/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).rename({"source1_entity_id": "s1_id"})
m = m.filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(",").alias("s23_id")).explode("s23_id").select("s1_id", "s23_id")
s1t = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
af = prep("test", m.join(s1t.filter(pl.col("country") == "France"), on="s1_id").select("s1_id", "s23_id"))
n = {c: s1v.filter(pl.col("country") == c).height for c in ("US", "India")}; nf = s1t.filter(pl.col("country") == "France").height
res = None
for c in ("US", "India"):
    t = tv.filter(pl.col("country") == c).group_by(K).agg((pl.len() / n[c] * 1000).round(1).alias(f"{c}_true"))
    a = av.filter(pl.col("country") == c).group_by(K).agg((pl.len() / n[c] * 1000).round(1).alias(f"{c}_acc"), (pl.len() * (1 - pl.col("label").mean()) / n[c] * 1000).round(1).alias(f"{c}_fp"))
    r = t.join(a, on=K, how="full", coalesce=True)
    res = r if res is None else res.join(r, on=K, how="full", coalesce=True)
f = af.group_by(K).agg((pl.len() / nf * 1000).round(1).alias("FR_acc"))
res = res.join(f, on=K, how="full", coalesce=True).fill_null(0).sort("US_true", descending=True)
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
print(res)
print("totals/K", res.select(pl.exclude(K).sum()))
print("S1 with 0 matches: FR", 1 - af["s1_id"].n_unique() / nf, " US true", 1 - tv.filter(pl.col("country") == "US")["s1_id"].n_unique() / n["US"], " IN true", 1 - tv.filter(pl.col("country") == "India")["s1_id"].n_unique() / n["India"])
