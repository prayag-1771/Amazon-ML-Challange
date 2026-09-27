import numpy as np, polars as pl, time
from src.features import load_side, pair_features, context_features, blocking_feats
from src.prune import PRUNE_FEATS
from src.model import feature_cols, train, predict, tune_threshold, assign
from src.metric import macro_f05
from src.io_utils import load_ground_truth
from src.config import is_valid_expr
s1, q = load_side("train")
gt = load_ground_truth()
# synthetic candidates: true pair + 5 random S1 of the same country for 20k queries
qs = q.sample(20000, seed=0).select("q_row", "entity_id", "country")
s1i = s1.select("s1_row", pl.col("entity_id").alias("s1_id"), "country")
true = qs.rename({"entity_id": "s23_id"}).join(gt, on="s23_id", how="left").join(s1i.drop("country"), on="s1_id", how="left")
rng = np.random.default_rng(0)
rand = qs.with_columns(pl.Series("s1_row", rng.integers(0, len(s1), len(qs)).astype(np.uint32)))
cand = pl.concat([true.filter(pl.col("s1_row").is_not_null()).select(pl.col("q_row").cast(pl.UInt32), pl.col("s1_row").cast(pl.UInt32)),
                  rand.select(pl.col("q_row").cast(pl.UInt32), "s1_row")]).unique()
cand = cand.with_columns(pl.Series("tf_sim", rng.random(len(cand), dtype=np.float32)), pl.lit(None, pl.Int16).alias("tf_rank"),
                         pl.Series("emb_sim", rng.random(len(cand), dtype=np.float32)), pl.lit(0, pl.Int16).alias("emb_rank"))
cand = blocking_feats(cand).with_columns(pl.len().over("q_row").cast(pl.Float32).alias("q_ncand0"), pl.lit(0.5, pl.Float32).alias("p0"))
df = pair_features(cand, s1, q)
df = context_features(df, "p0"); df = context_features(df.drop("q_ncand", "s1_nq"), "nc_tset")
df = df.join(s1i.select(pl.col("s1_row").cast(pl.UInt32), "s1_id"), on="s1_row").join(q.select(pl.col("q_row").cast(pl.UInt32), pl.col("entity_id").alias("s23_id")), on="q_row")
df = df.join(gt.with_columns(pl.lit(1, pl.Int8).alias("label")), on=["s1_id", "s23_id"], how="left").with_columns(pl.col("label").fill_null(0))
feats = feature_cols(df); print(len(feats), feats)
tr = df.filter(~is_valid_expr()); va = df.filter(is_valid_expr())
m = train(tr, feats, rounds=100, valid=va)
va = va.with_columns(pl.Series("p", predict(m, va, feats)))
t, _ = tune_threshold(va, va["s1_id"].unique(), gt, grid=[0.3, 0.5, 0.7])
print("smoke OK", t, len(assign(va, t)))
