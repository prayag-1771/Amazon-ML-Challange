import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
exec(open("../work/v10/legal.py").read().split("v = addleg(")[0])
vs = pl.read_parquet(W + "valid_scores_stage2.parquet")
v = addleg(st("train", vs, True).filter(is_valid_expr("s1_id")), "train").filter(pl.col("st") & (pl.col("leg") == "add"))
print(v.with_columns(pl.col("p2").cut([.02, .1, .3, .5, .75, .9]).alias("b")).group_by("country", "b").agg(pl.len(), pl.col("label").mean().round(3)).sort("country", "b"))
s1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "business_name", "business_address"]).rename({"entity_id": "s1_id", "business_name": "b1", "business_address": "ad1"})
q = pl.concat([pl.read_parquet(W + f"norm_train_s{i}.parquet", columns=["entity_id", "business_name", "business_address"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "business_name": "bq", "business_address": "adq"})
x = v.filter((pl.col("country") == "US") & (pl.col("p2") < 0.75) & (pl.col("p2") > 0.05)).join(s1, on="s1_id").join(q, on="s23_id")
for r in x.sample(min(8, x.height), seed=1).iter_rows(named=True):
    print(f"\n# label={r['label']} p2={r['p2']:.3f} Q {r['bq']} | {r['adq']}  -> S1 {r['b1']} | {r['ad1']}")
    for c in vs.filter(pl.col("s1_id") == r["s1_id"]).sort("p2", descending=True).head(6).join(q, on="s23_id").iter_rows(named=True):
        print(f"    lab={c['label']} p2={c['p2']:.3f} {c['bq']} | {c['adq']}")
    g = x.filter(False)
