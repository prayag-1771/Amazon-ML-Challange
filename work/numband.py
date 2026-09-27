import sys
sys.path.insert(0, ".")
import polars as pl
from src.model import load_model, predict

W = "../work/"
bins = [0.1, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
def nums(split):
    n1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "nums"]).rename({"entity_id": "s1_id", "nums": "na"})
    nq = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "nums", "addr_empty"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "nums": "nb"})
    return n1, nq
def add(d, n1, nq):
    d = d.join(n1, on="s1_id").join(nq, on="s23_id")
    return d.with_columns(
        pl.when((pl.col("na") == "") | (pl.col("nb") == "")).then(None)
        .otherwise(pl.col("nb").str.split(" ").list.contains(pl.col("na").str.split(" ").list.first()).cast(pl.Float32)).alias("numeq"),
        pl.col("p").cut(bins).alias("b"))

model, feats, T = load_model("lgb_v3")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats)))
va = va.sort("p", descending=True).unique("s23_id", keep="first")
va = add(va, *nums("train"))
pl.Config.set_tbl_rows(60)
print(va.group_by("country", "b", "label").agg(pl.len(), pl.col("numeq").mean(), pl.col("numeq").is_null().mean().alias("nonum")).sort("country", "b", "label"))
print(va.group_by("country", "b").agg(pl.len(), pl.col("label").mean(), pl.col("numeq").mean(), pl.col("addr_empty").mean()).sort("country", "b"))
te = pl.read_parquet(W + "test_scores_lgb_v3.parquet").sort("p", descending=True).unique("s23_id", keep="first")
te = add(te, *nums("test"))
print(te.group_by("country", "b").agg(pl.len(), pl.col("numeq").mean(), pl.col("addr_empty").mean()).sort("country", "b"))
