import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_source
exec(open("../work/v10/street.py").read().split("v = st(")[0])
W = "../work/"
def cell(split, sc):
    x = st(split, sc, False)
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "legal"]).rename({"entity_id": "s1_id", "legal": "l1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "legal"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "legal": "lq"})
    x = x.join(s1, on="s1_id").join(q, on="s23_id")
    l1, lq = pl.col("l1").fill_null(""), pl.col("lq").fill_null("")
    x = x.with_columns(pl.when(l1 == lq).then(pl.lit("same")).when(l1 == "").then(pl.lit("add")).when(lq == "").then(pl.lit("drop")).otherwise(pl.lit("chg")).alias("leg"))
    return x.filter((pl.col("u1") == pl.col("uq")) & (pl.col("p") >= .9))
T = {"US": .75, "India": .8, "France": .75}
v = cell("train", pl.read_parquet(W + "valid_scores_stage2.parquet")).filter(is_valid_expr("s1_id"))
v = v.with_columns(pl.col("ov").cut([.34, .67, .99]).alias("ob"))
print("valid (p>=.9, name=, nums=):")
print(v.group_by("country", "ob", "leg").agg(pl.len(), pl.col("label").mean().round(3).alias("prec"), (pl.col("p2") < pl.col("country").replace_strict(T)).sum().alias("nrej"),
      pl.col("label").filter(pl.col("p2") < pl.col("country").replace_strict(T)).mean().round(3).alias("prec_rej")).sort("country", "ob", "leg"))
sc = pl.read_parquet(W + "test_scores_stage2_v11.parquet").join(pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id")
t = cell("test", sc).filter(pl.col("country") == "France").with_columns(pl.col("ov").cut([.34, .67, .99]).alias("ob"))
print("France:"); print(t.group_by("ob", "leg").agg(pl.len(), (pl.col("p2") < .75).sum().alias("nrej"), pl.col("p2").filter(pl.col("p2") < .75).median().round(3)).sort("ob", "leg"))
t.write_parquet(W + "v10/nnst_fr.parquet")
