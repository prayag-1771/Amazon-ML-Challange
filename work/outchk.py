"""Label-free check of a matching_results.tsv: accepted up-shift pairs in excess of R * accepted down-shift pairs."""
import sys; sys.path.insert(0, ".")
import polars as pl
import src.stage2 as s2
R = pl.read_parquet("../work/shift_R.parquet"); Rp = pl.read_parquet("../work/shift_Rp.parquet")
c = pl.read_parquet("../work/norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
for f in sys.argv[1:]:
    m = pl.read_csv(f, separator="\t").rename(lambda x: x.lower())
    idc, lc = m.columns[0], m.columns[1]
    pairs = m.select(pl.col(idc).cast(pl.Utf8).alias("s1_id"), pl.col(lc).cast(pl.Utf8).str.split(",").alias("s23_id")).explode("s23_id").filter(pl.col("s23_id").str.len_chars() > 0)
    pairs = pairs.join(c, on="s1_id")
    x = s2.shift_cells(pairs, "test").filter(pl.col("dg").is_not_null())
    x = x.join(R, on=["country", "dg"], how="left").join(Rp.rename({"R": "Rp"}), on="dg", how="left").with_columns(pl.coalesce("R", "Rp").alias("R"))
    g = x.group_by("country", "dg").agg(pl.col("up").sum().alias("au"), (~pl.col("up")).sum().alias("ad"), pl.col("R").first())
    e = g.group_by("country").agg((pl.col("au") - pl.col("R") * pl.col("ad")).sum().round(0).alias("excess_up")).sort("country")
    print(f, "pairs", pairs.group_by("country").len().sort("country").rows(), "excess", e.rows())
