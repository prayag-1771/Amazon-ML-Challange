import sys
sys.path.insert(0, ".")
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from src.io_utils import load_ground_truth
from src.config import is_valid_expr
W = "../work/"
gt = load_ground_truth()
cand = pl.read_parquet(W + "pruned_train.parquet", columns=["s1_id", "s23_id"])
full = pl.read_parquet(W + "cand_train.parquet", columns=["q_row", "s1_row"])
n1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country", "name_core", "name_full", "state", "addr_empty"]).rename({"entity_id": "s1_id"})
n1 = n1.with_columns(pl.len().over("country", "name_core").alias("dup"))
nq = pl.concat([pl.read_parquet(W + f"norm_train_s{i}.parquet", columns=["entity_id", "name_core", "state", "addr_empty", "business_name"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "qc", "state": "qs", "addr_empty": "qe"})
gtv = gt.filter(is_valid_expr("s1_id"))
miss = gtv.join(cand, on=["s1_id", "s23_id"], how="anti").join(n1, on="s1_id").join(nq, on="s23_id")
miss = miss.with_columns(pl.Series("r", cpdist(miss["name_core"].to_list(), miss["qc"].to_list(), scorer=fuzz.ratio, workers=-1)))
miss = miss.with_columns((pl.col("name_core") == pl.col("qc")).alias("eq"),
                         pl.col("r").cut([50, 70, 85, 99.9]).alias("rb"),
                         pl.col("dup").cut([1, 3, 10, 50]).alias("db"))
pl.Config.set_tbl_rows(60)
print("misses", len(miss))
print(miss.group_by("country", "qe", "eq").len().sort("country", "qe", "eq"))
print(miss.group_by("country", "rb").len().sort("country", "rb"))
print(miss.filter(pl.col("eq")).group_by("country", "db").len().sort("country", "db"))
print(miss.filter(pl.col("eq")).group_by("country", (pl.col("state") == pl.col("qs")).alias("st_eq")).len())
# for eq-name misses: how many S1 with the same name AND same state
print(miss.filter(pl.col("eq") & (pl.col("qe") == 1)).select("country", "name_core", "dup", "state", "qs", "business_name").sample(30, seed=1))
