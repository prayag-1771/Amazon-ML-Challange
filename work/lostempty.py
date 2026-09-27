import sys
sys.path.insert(0, ".")
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from src.config import is_valid_expr
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
W = "../work/"
s1all = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country", "name_core"]).rename({"entity_id": "s1_id", "name_core": "n1"})
cnt = s1all.group_by("country", "n1").len("n1_cnt")
s1v = s1all.filter(is_valid_expr("s1_id"))
gt = pl.read_parquet(W + "gt_pairs.parquet").join(s1v, on="s1_id")
pr = pl.read_parquet(W + "pruned_train.parquet", columns=["s1_id", "s23_id"])
q = pl.concat([pl.read_parquet(W + f"norm_train_s{k}.parquet", columns=["entity_id", "name_core", "addr_empty"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq"})
d = gt.join(pr, on=["s1_id", "s23_id"], how="anti").join(q, on="s23_id").join(cnt, on=["country", "n1"])
d = d.with_columns(pl.Series("sim", cpdist(d["n1"].to_list(), d["nq"].to_list(), scorer=fuzz.token_sort_ratio, workers=-1)))
# how many S1 in country share query's exact name_core
qc = cnt.rename({"n1": "nq", "n1_cnt": "q_in_s1"})
d = d.join(qc, on=["country", "nq"], how="left").with_columns(pl.col("q_in_s1").fill_null(0))
print(d.group_by("country", "addr_empty").agg(pl.len(), (pl.col("n1_cnt") == 1).mean().alias("true_name_unique"),
      (pl.col("nq") == pl.col("n1")).mean().alias("name_core_eq"), (pl.col("sim") >= 90).mean().alias("sim90"),
      (pl.col("q_in_s1") == 0).mean().alias("qname_absent")).sort("country", "addr_empty"))
x = d.filter(pl.col("addr_empty") == 1)
print(x.group_by(pl.col("n1_cnt").clip(upper_bound=10)).len().sort("n1_cnt"))
