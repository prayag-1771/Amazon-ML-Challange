import polars as pl
from src.io_utils import load_ground_truth
from src.config import is_valid_expr

W = "../work/"
cand = pl.read_parquet(W + "pruned_train.parquet", columns=["s1_id", "s23_id"])
full = pl.read_parquet(W + "cand_train.parquet", n_rows=1)
print(full.schema)
gt = load_ground_truth()
raw1 = pl.read_parquet(W + "raw_train_s1.parquet").select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"), "country")
raw23 = pl.concat([pl.read_parquet(W + f"raw_train_s{i}.parquet") for i in (2, 3)]).select(pl.col("entity_id").alias("s23_id"), pl.col("business_name").alias("nq"), pl.col("business_address").alias("aq"))
miss = gt.join(cand, on=["s1_id", "s23_id"], how="anti").join(raw1, on="s1_id").join(raw23, on="s23_id")
# does the query have any pruned candidate at all?
hasc = cand.select("s23_id").unique().with_columns(pl.lit(True).alias("has_cand"))
miss = miss.join(hasc, on="s23_id", how="left").with_columns(pl.col("has_cand").fill_null(False))
print(miss.group_by("country", "has_cand").len())
miss = miss.with_columns((pl.col("aq").fill_null("").str.strip_chars() == "").alias("q_noaddr"),
                         pl.col("nq").str.contains(r"[^\x00-\x7F]").alias("q_nonascii"))
print(miss.group_by("country").agg(pl.len(), pl.col("q_noaddr").mean(), pl.col("q_nonascii").mean()))
for c in ("India", "US"):
    print("=" * 20, c)
    for r in miss.filter(pl.col("country") == c).sample(35, seed=3).iter_rows(named=True):
        print(f"Q : {r['nq']} | {r['aq']}   [has_cand={r['has_cand']}]")
        print(f"S1: {r['n1']} | {r['a1']}")
