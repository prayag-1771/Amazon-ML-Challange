import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
exec(open("../work/v10/street.py").read().split("v = st(")[0].split("W = ")[1].join(["W = ", ""]) if False else open("../work/v10/street.py").read().split("v = st(")[0])
def addleg(x, split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "legal"]).rename({"entity_id": "s1_id", "legal": "l1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "legal"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "legal": "lq"})
    x = x.join(s1, on="s1_id").join(q, on="s23_id")
    l1, lq = pl.col("l1").fill_null(""), pl.col("lq").fill_null("")
    return x.with_columns(pl.when(l1 == lq).then(pl.lit("same")).when(l1 == "").then(pl.lit("add")).when(lq == "").then(pl.lit("drop")).otherwise(pl.lit("chg")).alias("leg"),
                          (pl.col("ov") > 0.34).alias("st"))
v = addleg(st("train", pl.read_parquet(W + "valid_scores_stage2.parquet"), True).filter(is_valid_expr("s1_id")), "train")
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id"))
T = {"US": .75, "India": .8}
for c in ("US", "India"):
    n = (s1v["country"] == c).sum()
    print(c); print(v.filter(pl.col("country") == c).group_by("st", "leg").agg((pl.len() / n * 1000).round(1).alias("perK"), pl.col("label").mean().round(3).alias("prec"),
        (pl.col("p2") >= T[c]).mean().round(3).alias("accr"), pl.col("label").filter(pl.col("p2") < T[c]).mean().round(3).alias("prec_rej"), (pl.col("p2") < T[c]).sum().alias("nrej")).sort("st", "leg"))
t = addleg(pl.read_parquet("../work/v10/fr_nameeq.parquet").drop("country"), "test")
nf = 55680
print("France"); print(t.group_by("st", "leg").agg(pl.len().alias("n"), pl.col("acc").mean().round(3).alias("accr"), (~pl.col("acc")).sum().alias("nrej"), pl.col("p2").filter(~pl.col("acc")).median().round(3).alias("rejp2")).sort("st", "leg"))
