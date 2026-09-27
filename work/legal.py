import sys
sys.path.insert(0, ".")
import polars as pl
from src.model import load_model, predict
from src.run_pipeline import featurize, add_ce
from src.prune import prune

W = "../work/"
model, feats, T = load_model("lgb_v3")
va = pl.read_parquet(W + "valid_feats.parquet")
va = va.with_columns(pl.Series("p", predict(model, va, feats)))
s1 = pl.read_parquet(W + "raw_train_s1.parquet").select(pl.col("entity_id").alias("s1_id"), "country")
va = va.join(s1, on="s1_id")
key = ["nc_eq", "legal_eq", "num_first_eq", "q_addr_empty"]
g = va.filter(pl.col("p0_qrank") == 1).group_by(["country"] + key).agg(
    pl.len(), pl.col("label").mean().alias("lab"), pl.col("p").mean().alias("p"), pl.col("ce_logit").mean().alias("ce"))
pl.Config.set_tbl_rows(100)
print(g.filter(pl.col("len") > 2000).sort(["country"] + key))

# test France: same buckets on a sample of queries
cand = prune("test")
s1t = pl.read_parquet(W + "raw_test_s1.parquet").select(pl.col("entity_id").alias("s1_id"), "country")
cand = cand.join(s1t, on="s1_id")
qs = cand.select("q_row", "country").unique("q_row").group_by("country").agg(pl.col("q_row").sample(150_000, seed=0)).explode("q_row")
cs = cand.join(qs.select("q_row"), on="q_row", how="semi").drop("country")
df = add_ce(featurize(cs, "test"), "test")
df = df.with_columns(pl.Series("p", predict(model, df, feats))).join(s1t, on="s1_id")
df.write_parquet(W + "test_sample_feats.parquet")
g = df.filter(pl.col("p0_qrank") == 1).group_by(["country"] + key).agg(
    pl.len(), pl.col("p").mean().alias("p"), (pl.col("p") > T).mean().alias("assigned"), pl.col("ce_logit").mean().alias("ce"))
print(g.filter(pl.col("len") > 500).sort(["country"] + key))
