import sys
sys.path.insert(0, ".")
import polars as pl
W = "../work/"
for sp in ("train", "test"):
    s1 = pl.read_parquet(W + f"norm_{sp}_s1.parquet", columns=["entity_id", "country", "name_core", "addr_core", "nums", "addr_empty"])
    city = pl.read_parquet(W + f"norm_{sp}_s1.parquet", columns=["alpha_comps"])
    s1 = s1.with_columns(pl.len().over("country", "name_core").alias("ncnt"))
    print(f"== {sp} S1 name_core repeat count distribution (share of S1 with ncnt bucket)")
    print(s1.group_by("country").agg(
        (pl.col("ncnt") == 1).mean().alias("uniq"), (pl.col("ncnt").is_between(2, 4)).mean().alias("2-4"),
        (pl.col("ncnt").is_between(5, 20)).mean().alias("5-20"), (pl.col("ncnt") > 20).mean().alias(">20"),
        pl.col("addr_empty").mean().alias("addr_empty"), (pl.col("nums").fill_null("") == "").mean().alias("no_num"),
        pl.col("name_core").str.split(" ").list.len().mean().alias("ntok")).sort("country"))
    for i in (2, 3):
        q = pl.read_parquet(W + f"norm_{sp}_s{i}.parquet", columns=["country", "name_core", "addr_empty", "nums", "state"])
        print(sp, "S%d" % i, q.group_by("country").agg(pl.col("addr_empty").mean().alias("addr_empty"),
              (pl.col("nums").fill_null("") == "").mean().alias("no_num"), (pl.col("state").fill_null("") == "").mean().alias("no_state"),
              pl.col("name_core").str.split(" ").list.len().mean().alias("ntok")).sort("country").rows())
