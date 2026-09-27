"""Build v16 TSVs: v15 outputs, India addressed channel replaced by `ind` (s1_id,s23_id parquet), plus extra adds."""
import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from pathlib import Path
from src.io_utils import write_id_lists
ind_new, extras, out = sys.argv[1], sys.argv[2:-1], Path(sys.argv[-1])
def rd(p, c):
    t = pl.read_csv(p, separator="\t", schema_overrides={c: pl.Utf8}).with_columns(pl.col(c).fill_null(""))
    return t["source1_entity_id"].to_list(), t.with_columns(pl.col(c).str.split(",")).explode(c).filter(pl.col(c) != "") \
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col(c).alias("s23_id"))
s1_ids, m = rd("sub15_indiaaddr/matching_results.tsv", "matched_entity_ids")
_, c = rd("sub15_indiaaddr/candidate_pairs.tsv", "candidate_entity_ids")
old = pl.read_parquet("india_addr_test_accept.parquet").select("s1_id", "s23_id")
new = pl.read_parquet(ind_new).select("s1_id", "s23_id")
print("v15", m.height, c.height, "old ind", old.height, "new ind", new.height)
m = m.join(old, on=["s1_id", "s23_id"], how="anti"); c = c.join(old, on=["s1_id", "s23_id"], how="anti")
adds = [new] + [pl.read_parquet(e).select("s1_id", "s23_id") for e in extras]
for a in adds[1:]:
    print("extra", a.height, "overlap with matched s23", a.join(m, on="s23_id", how="semi").height)
m = pl.concat([m] + adds).unique(); c = pl.concat([c] + adds).unique()
print("v16", m.height, c.height, f"{c.height/len(s1_ids):.3f}/S1")
write_id_lists(out / "matching_results.tsv", s1_ids, m, "matched_entity_ids")
write_id_lists(out / "candidate_pairs.tsv", s1_ids, c, "candidate_entity_ids")
