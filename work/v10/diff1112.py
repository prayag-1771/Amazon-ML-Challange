import polars as pl
def load(p):
    d = pl.read_csv(p, separator="\t"); c = d.columns
    return d.with_columns(pl.col(c[1]).str.split(",")).explode(c[1]).drop_nulls().filter(pl.col(c[1]) != "").rename({c[0]: "s1_id", c[1]: "s23_id"})
a = load("../work/probe11_fr_vfix/matching_results.tsv"); b = load("../work/sub12_cascade/matching_results.tsv")
s1 = pl.read_parquet("../work/norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
a = a.join(s1, on="s1_id"); b = b.join(s1, on="s1_id")
for c in ("US", "India", "France"):
    A = a.filter(pl.col("country") == c).select("s1_id", "s23_id"); B = b.filter(pl.col("country") == c).select("s1_id", "s23_id")
    print(c, len(A), len(B), "only11", len(A.join(B, on=["s1_id", "s23_id"], how="anti")), "only12", len(B.join(A, on=["s1_id", "s23_id"], how="anti")), flush=True)
b.write_parquet("../work/v10/v12_matches.parquet")
