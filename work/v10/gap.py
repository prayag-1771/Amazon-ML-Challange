"""All top-1 pairs per category (name rel x num edit x token kind): perK and precision on val US/India vs France perK and
vfix acceptance. Expected France true/K = FR_perK * US_prec (or IN); gap = expected - FR_acc -> where France recall is lost."""
import sys
sys.path.insert(0, ".")
import polars as pl
from rapidfuzz.distance import Levenshtein as L
from src.config import is_valid_expr
W = "../work/"
src = open("../work/v10/frcat.py").read()
exec(src[src.index("def cats"):src.index("def load")])
exec(open("../work/v10/numedit.py").read().split("def prep")[0].split("exec(src")[1].split("\n", 1)[1])
VAR = ["center", "services", "service", "partners", "fils", "groupe", "associes", "developpement", "france"]
def prep(split, sc):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    x = cats(sc.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id"))
    t1 = pl.col("n1").fill_null("").str.split(" "); tq = pl.col("nq").fill_null("").str.split(" ")
    x = x.with_columns(t1.list.set_difference(tq).list.first().alias("dt"), tq.list.set_difference(t1).list.first().alias("at"))
    x = x.with_columns(pl.struct("dt", "at").map_elements(lambda s: L.normalized_similarity(s["dt"] or "", s["at"] or ""), return_dtype=pl.Float64).alias("sim"),
                       pl.struct("a1", "aq").map_elements(lambda s: etype(s["a1"], s["aq"]), return_dtype=pl.String).alias("e"))
    ocr = pl.col("at").str.contains(r"\d") | (pl.col("at").str.len_chars() <= 2)
    x = x.with_columns(pl.when(~pl.col("nm").is_in(["swap1", "addonly"])).then(pl.col("nm"))
        .when(pl.col("at").is_in(VAR)).then(pl.col("nm") + "_var").when(ocr | (pl.col("sim") >= 0.5)).then(pl.col("nm") + "_typo").otherwise(pl.col("nm") + "_word").alias("k"))
    return x.with_columns(pl.col("e").replace({"sub±1+": "near+", "sub+": "near+", "sub±1-": "near-", "sub-": "near-", "sub_far": "far"}).alias("e"))
K = ["k", "e"]
v = prep("train", pl.read_parquet(W + "valid_scores_stage2.parquet"))
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id"))
res = None
for c in ("US", "India"):
    n = s1v.filter(pl.col("country") == c).height
    r = v.filter(pl.col("country") == c).group_by(K).agg((pl.len() / n * 1000).round(1).alias(f"{c[:2]}_K"), pl.col("label").mean().round(3).alias(f"{c[:2]}_prec"))
    res = r if res is None else res.join(r, on=K, how="full", coalesce=True)
t = prep("test", pl.read_parquet(W + "test_scores_stage2_v11.parquet"))
m = pl.read_csv(W + "probe11_fr_vfix/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).rename({"source1_entity_id": "s1_id"})
m = m.filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(",").alias("s23_id")).explode("s23_id").select("s1_id", "s23_id").with_columns(pl.lit(True).alias("acc"))
t = t.join(m, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("acc").fill_null(False))
nf = (pl.read_parquet(W + "norm_test_s1.parquet", columns=["country"])["country"] == "France").sum()
f = t.filter(pl.col("country") == "France").group_by(K).agg((pl.len() / nf * 1000).round(1).alias("FR_K"), pl.col("acc").mean().round(3).alias("FR_acc"),
      pl.col("p2").filter(~pl.col("acc")).median().round(3).alias("FR_rej_p2med"))
res = res.join(f, on=K, how="full", coalesce=True).fill_null(0).with_columns(
    (pl.col("FR_K") * (pl.col("FR_acc") - pl.col("US_prec"))).round(1).alias("gapUS"), (pl.col("FR_K") * (pl.col("FR_acc") - pl.col("In_prec"))).round(1).alias("gapIN"))
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
print(res.sort(pl.col("gapUS").abs(), descending=True))
res.write_parquet("../work/v10/gap.parquet")
t.filter(pl.col("country") == "France").select("s1_id", "s23_id", "p2", "acc", "k", "e", "n1", "nq", "a1", "aq", "dt", "at").write_parquet("../work/v10/fr_top1.parquet")
