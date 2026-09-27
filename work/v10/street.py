"""name= & num= pairs: split by street-token overlap (addr_core minus numbers and city comps)."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W = "../work/"
def st(split, sc, lab):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums", "addr_core", "alpha_comps"]).rename(
        {"entity_id": "s1_id", "name_core": "n1", "nums": "u1", "addr_core": "a1", "alpha_comps": "c1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums", "addr_core", "alpha_comps"]) for i in (2, 3)]).rename(
        {"entity_id": "s23_id", "name_core": "nq", "nums": "uq", "addr_core": "aq", "alpha_comps": "cq"})
    x = sc.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
    x = x.filter((pl.col("n1") == pl.col("nq")) & (pl.col("u1").str.split(" ").list.first() == pl.col("uq").str.split(" ").list.first()))
    tok = lambda a, c: pl.col(a).fill_null("").str.split(" ").list.set_difference(pl.col(c).fill_null("").str.replace_all(r"\|", " ").str.split(" ")).list.eval(pl.element().filter(~pl.element().str.contains(r"^\d")))
    x = x.with_columns(tok("a1", "c1").alias("s1t"), tok("aq", "cq").alias("sqt"))
    x = x.with_columns((pl.col("s1t").list.set_intersection("sqt").list.len() / pl.max_horizontal(pl.col("s1t").list.len(), pl.col("sqt").list.len(), 1)).alias("ov"),
                       (pl.col("aq").fill_null("") == "").alias("qe"))
    return x.with_columns(pl.when(pl.col("qe")).then(pl.lit("qempty")).otherwise(pl.col("ov").cut([0, 0.34, 0.67, 0.99]).cast(pl.String)).alias("ovb"))
v = st("train", pl.read_parquet(W + "valid_scores_stage2.parquet"), True).filter(is_valid_expr("s1_id"))
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id"))
m = pl.read_csv(W + "probe11_fr_vfix/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).rename({"source1_entity_id": "s1_id"})
m = m.filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(",").alias("s23_id")).explode("s23_id").select("s1_id", "s23_id", pl.lit(True).alias("acc"))
t = st("test", pl.read_parquet(W + "test_scores_stage2_v11.parquet"), False).filter(pl.col("country") == "France").join(m, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("acc").fill_null(False))
nf = (pl.read_parquet(W + "norm_test_s1.parquet", columns=["country"])["country"] == "France").sum()
T = {"US": .75, "India": .8}
for c in ("US", "India"):
    n = (s1v["country"] == c).sum()
    print(c); print(v.filter(pl.col("country") == c).group_by("ovb").agg((pl.len() / n * 1000).round(1).alias("perK"), pl.col("label").mean().round(3).alias("prec"),
        (pl.col("p2") >= T[c]).mean().round(3).alias("accr"), pl.col("label").filter(pl.col("p2") >= T[c]).mean().round(3).alias("prec_acc"),
        pl.col("label").filter(pl.col("p2") < T[c]).mean().round(3).alias("prec_rej")).sort("ovb"))
print("France"); print(t.group_by("ovb").agg((pl.len() / nf * 1000).round(1).alias("perK"), pl.col("acc").mean().round(3).alias("accr"), pl.col("p2").median().round(3)).sort("ovb"))
t.write_parquet("../work/v10/fr_nameeq.parquet")
