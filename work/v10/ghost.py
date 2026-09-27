"""Ghost-entity check: accepted pairs whose query house number conflicts with the S1's but agrees with other queries."""
import sys
sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W = "../work/"
def load(split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename(
        {"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename(
        {"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    return s1.with_columns(pl.col("u1").str.split(" ").list.first().alias("a1")), q.with_columns(pl.col("uq").str.split(" ").list.first().alias("aq"))
def study(top, s1, q, tag):
    x = top.join(s1, on="s1_id").join(q, on="s23_id")
    # number relation
    x = x.with_columns(
        pl.when((pl.col("a1").fill_null("") == "") | (pl.col("aq").fill_null("") == "")).then(pl.lit("miss"))
        .when(pl.col("a1") == pl.col("aq")).then(pl.lit("eq")).otherwise(pl.lit("diff")).alias("nrel"))
    # support: other accepted queries on the same S1 with the same query number
    x = x.with_columns(pl.len().over("s1_id", "aq").alias("same_aq_on_s1"))
    # all queries (any S1) with same name_core and same number (cluster size in query space)
    cl = q.filter(pl.col("aq").fill_null("") != "").group_by("nq", "aq").len().rename({"len": "q_clu"})
    x = x.join(cl, on=["nq", "aq"], how="left").with_columns(pl.col("q_clu").fill_null(1))
    print(f"== {tag}")
    g = x.group_by("country", "nrel").agg(pl.len(), *([pl.col("label").mean().alias("prec")] if "label" in x.columns else []))
    print(g.sort("country", "nrel"))
    d = x.filter(pl.col("nrel") == "diff").with_columns(
        pl.when(pl.col("q_clu") >= 2).then(pl.lit("clu2+")).otherwise(pl.lit("clu1")).alias("k"))
    g = d.group_by("country", "k").agg(pl.len(), *([pl.col("label").mean().alias("prec")] if "label" in x.columns else []))
    print(g.sort("country", "k"))
    return x
# test
s1, q = load("test")
d = pl.read_parquet(W + "test_scores_stage2_v9.parquet")
top = d.sort("p2", descending=True).unique("s23_id", keep="first")
T = {"US": 0.75, "India": 0.80, "France": 0.75}
c = s1.select("s1_id", "country")
top = top.join(c, on="s1_id").filter(pl.col("p2") >= pl.col("country").replace_strict(T)).drop("country")
xt = study(top, s1, q, "TEST accepted")
nS1 = c["country"].value_counts()
print(nS1)
xt.filter((pl.col("country") == "France") & (pl.col("nrel") == "diff")).sample(40, seed=1).select("n1", "u1", "nq", "uq", "p2", "q_clu").write_csv(W + "v10/ghost_fr_diff.tsv", separator="\t")
# val
s1, q = load("train")
v = pl.read_parquet(W + "valid_scores_stage2_v9.parquet") if False else None
a = pl.read_parquet(W + "valid_scores_lgb_v5cf.parquet").rename({"p": "pa"})
b = pl.read_parquet(W + "valid_scores_lgb_v6k.parquet").drop("label")
v = a.join(b, on=["s1_id", "s23_id"]).with_columns(((pl.col("pa") + pl.col("p")) / 2).alias("p2"))
top = v.sort("p2", descending=True).unique("s23_id", keep="first").filter(pl.col("p2") >= 0.75)
study(top, s1, q, "VAL accepted (ensemble, t=.75)")
