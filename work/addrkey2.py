"""Exact-address pairs outside pruned candidates: name similarity distribution, truth rate (train), per-query view."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl
from rapidfuzz import process, fuzz
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(300); pl.Config.set_tbl_cols(20)
W = "../work/"
def run(split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "addr_core", "name_core"]).with_row_index("s1_row")
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "country", "addr_core", "name_core"]) for k in (2, 3)]).with_row_index("q_row")
    a1 = s1.filter(pl.col("addr_core").str.len_chars() > 5); aq = q.filter(pl.col("addr_core").str.len_chars() > 5)
    ak = aq.join(a1, on=["country", "addr_core"], suffix="_1").select(pl.col("q_row", "s1_row").cast(pl.UInt32), "country",
        pl.col("entity_id").alias("s23_id"), pl.col("entity_id_1").alias("s1_id"), pl.col("name_core").alias("nq"), pl.col("name_core_1").alias("n1"))
    pr = pl.read_parquet(W + f"pruned_{split}.parquet", columns=["q_row", "s1_row"])
    ak = ak.join(pr.with_columns(pl.lit(True).alias("in_pr")), on=["q_row", "s1_row"], how="left").with_columns(pl.col("in_pr").fill_null(False))
    ak = ak.with_columns(pl.Series("ns", fuzz_ts(ak["nq"].to_list(), ak["n1"].to_list())))
    ak = ak.with_columns(pl.col("ns").cut([50, 70, 85, 95, 99.9]).alias("b"))
    if split == "train":
        gt = pl.read_parquet(W + "gt_pairs.parquet").with_columns(pl.lit(1).alias("y"))
        ak = ak.join(gt, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("y").fill_null(0))
        print(split, ak.group_by("country", "in_pr", "b").agg(pl.len(), pl.col("y").mean().alias("true")).sort("country", "in_pr", "b"))
    else:
        print(split, ak.group_by("country", "in_pr", "b").agg(pl.len(), pl.col("q_row").n_unique().alias("nq")).sort("country", "in_pr", "b"))
        m = ak.filter(~pl.col("in_pr") & (pl.col("ns") >= 85) & (pl.col("country") == "France"))
        print(m.sample(25, seed=1).select("nq", "n1", "ns"))
        m.select("q_row", "s1_row", "s23_id", "s1_id", "ns").write_parquet(W + "addrmiss_test.parquet")
def fuzz_ts(a, b):
    out = np.empty(len(a), np.float32)
    for i in range(0, len(a), 1_000_000):
        out[i:i+1_000_000] = process.cpdist(a[i:i+1_000_000], b[i:i+1_000_000], scorer=fuzz.token_set_ratio, workers=-1)
    return out
run("train"); run("test")
