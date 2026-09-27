"""Label-free precision proxy: first S1 house number found among query numbers, on assigned pairs.
Validation (with labels) calibrates proxy vs true precision; then apply to test by country and band."""
import sys
sys.path.insert(0, ".")
import polars as pl
from src.model import load_model, predict
from src.config import is_valid_expr
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
W = "../work/"
model, feats, T = load_model("lgb_v5cf")
def numeq(split, pairs):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "nums", "country", "name_core"]).rename({"entity_id": "s1_id", "nums": "n1", "name_core": "c1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "nums", "name_core"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "nums": "nq", "name_core": "cq"})
    d = pairs.join(s1, on="s1_id").join(q, on="s23_id")
    return d.with_columns(
        pl.when((pl.col("n1") == "") | (pl.col("nq") == "")).then(None).otherwise(
            pl.col("nq").str.split(" ").list.contains(pl.col("n1").str.split(" ").list.first()).cast(pl.Float64)).alias("numeq"),
        (pl.col("c1") == pl.col("cq")).cast(pl.Float64).alias("name_eq"),
        pl.col("p").cut([0.75, 0.9, 0.99, 0.999]).alias("band"))
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats)))
va = va.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= T).select("s1_id", "s23_id", "p", "label")
v = numeq("train", va)
print("VALID", v.group_by("country", "band").agg(pl.len(), pl.col("label").mean().alias("prec"), pl.col("numeq").mean(), pl.col("numeq").is_null().mean().alias("nonum"), pl.col("name_eq").mean()).sort("country", "band"))
print("VALID by label", v.group_by("country", "label").agg(pl.len(), pl.col("numeq").mean(), pl.col("name_eq").mean()).sort("country", "label"))
te = pl.read_parquet(W + "test_scores_lgb_v5cf.parquet").sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= T)
t = numeq("test", te)
print("TEST", t.group_by("country", "band").agg(pl.len(), pl.col("numeq").mean(), pl.col("numeq").is_null().mean().alias("nonum"), pl.col("name_eq").mean()).sort("country", "band"))
