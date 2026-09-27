"""Rescue model for empty-address queries: name-only char-3gram retrieval -> small LightGBM -> per-S1 F gain on validation."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl, lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from rapidfuzz.distance import JaroWinkler
from src.config import is_valid_expr
W = "../work/"
T = {"US": .75, "India": .8}
nm = pl.read_parquet(W+"namechan_train_empty.parquet").filter(pl.col("rk") <= 10)
s1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country","name_core","legal","addr_empty"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
q = pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet", columns=["entity_id","name_core","legal"]) for k in (2,3)]).with_row_index("q_row").with_columns(pl.col("q_row").cast(pl.UInt32))
s1 = s1.join(s1.group_by("country","name_core").len("s1_ncnt"), on=["country","name_core"])
qn = q.join(s1.select("s1_row","country").head(0), how="cross") if False else None
d = nm.join(s1.select("s1_row", pl.col("entity_id").alias("s1_id"), "country", pl.col("name_core").alias("n1"), pl.col("legal").alias("l1"), "s1_ncnt", pl.col("addr_empty").alias("s1_empty")), on="s1_row")
d = d.join(q.select("q_row", pl.col("entity_id").alias("s23_id"), pl.col("name_core").alias("nq"), pl.col("legal").alias("lq")), on="q_row")
d = d.join(d.group_by("country","nq").agg(pl.col("q_row").n_unique().alias("q_ncnt")), on=["country","nq"])
d = d.with_columns(
    (pl.col("sim") - pl.col("sim").max().over("q_row")).alias("gap_best"),
    (pl.col("sim") - pl.col("sim").top_k(2).min().over("q_row")).alias("gap_2nd"),
    (pl.col("sim") >= pl.col("sim").max().over("q_row") - 0.02).sum().over("q_row").alias("n_tie"),
    pl.len().over("q_row").alias("n_c"),
    (pl.col("n1") == pl.col("nq")).cast(pl.Int8).alias("eq"),
    pl.when((pl.col("l1") == "") | (pl.col("lq") == "")).then(-1).otherwise((pl.col("l1") == pl.col("lq")).cast(pl.Int8)).alias("leq"),
    (pl.col("country") == "India").cast(pl.Int8).alias("is_in"),
)
n1, n2 = d["n1"].to_list(), d["nq"].to_list()
d = d.with_columns(pl.Series("r", cpdist(n1, n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                   pl.Series("ts", cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                   pl.Series("jw", cpdist(n1, n2, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)),
                   pl.col("n1").str.len_chars().alias("len1"), pl.col("nq").str.len_chars().alias("lenq"))
d = d.with_columns((pl.col("r") - pl.col("r").max().over("q_row")).alias("r_gap"),
                   (pl.col("r") >= pl.col("r").max().over("q_row") - 3).sum().over("q_row").alias("r_ntie"),
                   pl.len().over("s1_row").alias("s1_nq"))
gt = pl.read_parquet(W+"gt_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("label"))
d = d.join(gt, on=["s1_id","s23_id"], how="left").with_columns(pl.col("label").fill_null(0), is_valid_expr("s1_id").alias("vs1"))
F = ["sim","rk","gap_best","gap_2nd","n_tie","n_c","eq","leq","is_in","s1_ncnt","q_ncnt","s1_empty","r","ts","jw","len1","lenq","r_gap","r_ntie","s1_nq"]
# train on queries touching no validation S1 in this channel
vq = d.filter(pl.col("vs1")).select("q_row").unique()
tr = d.join(vq, on="q_row", how="anti")
va = d.join(vq, on="q_row", how="semi")
print("train", len(tr), tr["label"].mean(), "valid", len(va), va["label"].mean(), flush=True)
m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, verbose=-1, num_threads=14),
              lgb.Dataset(tr.select(F).to_numpy().astype(np.float32), tr["label"].to_numpy()), 600)
m.save_model(W+"v10/rescue.lgb")
va = va.with_columns(pl.Series("pr", m.predict(va.select(F).to_numpy().astype(np.float32))))
va.write_parquet(W+"v10/rescue_valid.parquet")
print(sorted(zip(F, m.feature_importance("gain").round()), key=lambda x: -x[1]))
