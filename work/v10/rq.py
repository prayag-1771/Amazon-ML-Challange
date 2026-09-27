import sys; sys.path.insert(0, "../business_entity_resolution")
import polars as pl, numpy as np, lightgbm as lgb
from src import rescue
d = rescue.pairs("train")
d = d.with_columns(pl.Series("pr", lgb.Booster(model_file="rescue.lgb").predict(d.select(rescue.F).to_numpy().astype(np.float32))))
d.write_parquet("v10/rescue_train_pairs.parquet")
fn = pl.read_parquet("v10/val_fn15.parquet").filter((pl.col("addr_empty")==1)&(pl.col("cat")=="noquerycand"))
x = fn.join(d.select("s1_id","s23_id","rk","pr","eq","s1_ncnt","q_ncnt","n_tie","sim"), on=["s1_id","s23_id"], how="left")
for c in ("US","India"):
    y = x.filter(pl.col("country")==c)
    print(c, len(y), "retrieved", y["rk"].is_not_null().mean(), "rk1", (y["rk"]==1).mean(), "eq", (y["eq"]==1).mean())
    print(y.group_by(pl.col("eq"), pl.col("s1_ncnt").clip(0,5)).agg(pl.len(), (pl.col("rk")==1).mean().alias("rk1"), pl.col("pr").mean()).sort("len", descending=True).head(12))
