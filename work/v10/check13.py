"""Sanity-check a new submission dir vs v12: per-country diff, rescue coverage, matches ⊆ candidates, query uniqueness."""
import sys
import polars as pl
D = sys.argv[1]
def load(p):
    d = pl.read_csv(p, separator="\t"); c = d.columns
    return d.with_columns(pl.col(c[1]).str.split(",")).explode(c[1]).drop_nulls().filter(pl.col(c[1]) != "").rename({c[0]: "s1_id", c[1]: "s23_id"}).select("s1_id", "s23_id")
s1 = pl.read_parquet("norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
old = load("sub12_cascade/matching_results.tsv").join(s1, on="s1_id")
new = load(f"{D}/matching_results.tsv").join(s1, on="s1_id")
cand = load(f"{D}/candidate_pairs.tsv")
K = ["s1_id", "s23_id"]
for c in ("US", "India", "France"):
    A = old.filter(pl.col("country") == c).select(K); B = new.filter(pl.col("country") == c).select(K)
    print(f"{c:7s} v12 {len(A):,}  new {len(B):,}  only_v12 {len(A.join(B, on=K, how='anti')):,}  only_new {len(B.join(A, on=K, how='anti')):,}")
print("matches not in candidates:", len(new.select(K).join(cand, on=K, how="anti")))
print("queries matched to >1 S1:", new.group_by("s23_id").agg(pl.col("s1_id").n_unique().alias("k")).filter(pl.col("k") > 1).height)
print("S1 rows new/cand:", new["s1_id"].n_unique(), load(f"{D}/candidate_pairs.tsv")["s1_id"].n_unique(), "total S1", s1.height)
r = pl.read_parquet("v10/rescue_test_accept.parquet").select(K + ["country"])
hit = r.join(new.select(K), on=K, how="semi")
print("my rescue pairs present in new:", f"{hit.height:,} / {r.height:,}", hit.group_by("country").len().to_dicts())
extra = new.select(K).join(old.select(K), on=K, how="anti").join(r.select(K), on=K, how="anti")
print("new-only pairs not from my rescue:", f"{extra.height:,}")
