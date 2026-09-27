"""France legal-form-add accepts on top of probe11_fr_vfix: top-1, name_core equal, all house numbers equal,
street overlap > .34, S1 legal empty / query legal set, stage-1 p >= .9, rejected by stage 2 (p2 < .75)."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.io_utils import load_source, write_id_lists
from pathlib import Path
W = "../work/"
t = pl.read_parquet(W + "v10/legrule_fr.parquet").filter((pl.col("p") >= .9) & (pl.col("p2") < .75)).select("s1_id", "s23_id")
t.write_parquet(W + "v10/fr_legadd_accept.parquet")
m = pl.read_csv(W + "probe11_fr_vfix/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).rename({"source1_entity_id": "s1_id"})
acc = m.filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(",").alias("s23_id")).explode("s23_id").select("s1_id", "s23_id")
assert t.join(acc, on="s23_id").height == 0
allm = pl.concat([acc, t])
assert allm["s23_id"].is_unique().all()
out = Path(W + "probe12_fr_legadd"); out.mkdir(exist_ok=True)
write_id_lists(out / "matching_results.tsv", load_source("test", 1)["entity_id"].to_list(), allm, "matched_entity_ids")
print(f"vfix {acc.height:,} + legadd {t.height:,} = {allm.height:,}")
