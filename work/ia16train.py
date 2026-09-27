import sys, time; sys.path.insert(0,'../business_entity_resolution')
import lightgbm as lgb, numpy as np, polars as pl
from src import india_addr as ia
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
t0=time.time(); W=ia.W
gt=load_ground_truth()
hit=gt.join(pl.read_parquet(W+"pruned_train.parquet",columns=["s1_id","s23_id"]),on=["s1_id","s23_id"],how="semi")
d=ia.pairs("train",hit).join(gt.with_columns(pl.lit(1,pl.Int8).alias("label")),on=["s1_id","s23_id"],how="left")
d=d.with_columns(pl.col("label").fill_null(0),is_valid_expr("s1_id").alias("vs1"))
print("pairs",d.shape,time.time()-t0,flush=True)
vq=d.filter(pl.col("vs1")).select("q_row").unique()
tr,va=d.join(vq,on="q_row",how="anti"),d.join(vq,on="q_row",how="semi")
X,y=tr.select(ia.F).to_numpy().astype(np.float32),tr["label"].to_numpy()
for s in ia.SEEDS:
    lgb.train({**ia.P,"seed":s},lgb.Dataset(X,y),500).save_model(W+f"india_addr16_s{s}.lgb")
X=va.select(ia.F).to_numpy().astype(np.float32)
va=va.with_columns(pl.Series("pr",np.mean([lgb.Booster(model_file=W+f"india_addr16_s{s}.lgb").predict(X) for s in ia.SEEDS],axis=0)))
va.select("q_row","s1_id","s23_id","label","vs1","pr").write_parquet(W+"ia16_valid.parquet")
print("done",time.time()-t0,flush=True)
