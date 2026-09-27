"""same name/num/street top-1 pairs where the query changes legal form: does the query's legal agree with the legal forms of
the S1's other confidently-matched queries? Val precision by agreement, and France acceptance."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
exec(open("../work/v10/legal.py").read().split("v = addleg(")[0])
def sib(x, sc, split):
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "legal"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "legal": "ls"})
    top = sc.sort("p2", descending=True).unique("s23_id", keep="first").filter(pl.col("p2") >= 0.9).select("s1_id", "s23_id").join(q, on="s23_id")
    top = top.with_columns(pl.col("ls").fill_null(""))
    x = x.join(top.rename({"s23_id": "o"}), on="s1_id", how="left").filter(pl.col("o").is_null() | (pl.col("o") != pl.col("s23_id")))
    return x.group_by("s1_id", "s23_id").agg(pl.all().exclude("o", "ls").first(), (pl.col("ls") == pl.col("lq").fill_null("")).sum().alias("agree"),
        ((pl.col("ls") != pl.col("lq").fill_null("")) & (pl.col("ls") != "") & pl.col("ls").is_not_null()).sum().alias("disagree"), pl.col("o").count().alias("nsib"))
vs = pl.read_parquet(W + "valid_scores_stage2.parquet")
v = addleg(st("train", vs, True).filter(is_valid_expr("s1_id")), "train").filter(pl.col("st") & pl.col("leg").is_in(["add", "chg"]))
v = sib(v, vs, "train").with_columns(pl.when(pl.col("agree") > 0).then(pl.lit("agree")).when(pl.col("disagree") > 0).then(pl.lit("disagree")).otherwise(pl.lit("none")).alias("sg"))
T = {"US": .75, "India": .8}
for c in ("US", "India"):
    print(c); print(v.filter(pl.col("country") == c).group_by("leg", "sg").agg(pl.len(), pl.col("label").mean().round(3).alias("prec"),
        (pl.col("p2") >= T[c]).mean().round(3).alias("accr"), pl.col("label").filter(pl.col("p2") < T[c]).mean().round(3).alias("prec_rej"), (pl.col("p2") < T[c]).sum().alias("nrej")).sort("leg", "sg"))
ts = pl.read_parquet(W + "test_scores_stage2_v11.parquet")
t = addleg(pl.read_parquet("../work/v10/fr_nameeq.parquet").drop("country"), "test").filter(pl.col("st") & pl.col("leg").is_in(["add", "chg"]))
t = sib(t, ts, "test").with_columns(pl.when(pl.col("agree") > 0).then(pl.lit("agree")).when(pl.col("disagree") > 0).then(pl.lit("disagree")).otherwise(pl.lit("none")).alias("sg"))
print("France"); print(t.group_by("leg", "sg").agg(pl.len(), pl.col("acc").mean().round(3).alias("accr"), (~pl.col("acc")).sum().alias("nrej"), pl.col("p2").filter(~pl.col("acc")).median().round(3).alias("rejp2")).sort("leg", "sg"))
print(t.filter(~pl.col("acc")).with_columns(pl.col("p2").cut([.02, .1, .3, .5]).alias("b")).group_by("leg", "sg", "b").len().sort("leg", "sg", "b"))
t.write_parquet("../work/v10/fr_legsib.parquet")
