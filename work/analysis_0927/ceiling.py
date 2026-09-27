"""Test-side estimate of the irreducible loss: unassigned empty-address queries whose name_core is shared by >=2 S1
of the country. Calibrated with validation: loss per such query per S1 (val FN pairs -> F loss)."""
import polars as pl

W = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/"
# validation calibration (from lossdec.py): 13,912 ambiguous empty-address FN pairs over 220,567 S1 -> F loss 0.00566
# split by country: US 8,873 pairs / 132,085 S1 -> 0.00608 ; India 5,039 / 88,482 S1 -> 0.00503
val = {"US": (8873, 132085, 0.00608), "India": (5039, 88482, 0.00503)}
k_loss = {c: l / (n / s) for c, (n, s, l) in val.items()}  # F loss per (ambiguous FN pairs per S1)
k_pool = sum(l * s for n, s, l in val.values()) / sum(n for n, s, l in val.values())  # loss per pair * S1 count
print("F loss per ambiguous-FN-pair-per-S1:", {c: round(v, 3) for c, v in k_loss.items()}, "pooled", round(k_pool, 3))

m = pl.read_csv(W + "sub15_indiaaddr/matching_results.tsv", separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.Utf8})
assigned = m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").select(
    pl.col("matched_entity_ids").alias("s23_id")).filter(pl.col("s23_id") != "").unique()
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["country", "name_core"])
ns1 = s1.group_by("country").len("n_s1")
cnt = s1.group_by("country", "name_core").len("k")
q = pl.concat([pl.scan_parquet(W + f"norm_test_s{k}.parquet").select("entity_id", "country", "name_core", "addr_empty").filter(pl.col("addr_empty") == 1).collect()
               for k in (2, 3)]).rename({"entity_id": "s23_id"})
q = q.join(assigned, on="s23_id", how="anti").join(cnt, on=["country", "name_core"], how="left").with_columns(pl.col("k").fill_null(0))
r = q.group_by("country").agg(pl.len().alias("unassigned_empty"), (pl.col("k") >= 2).sum().alias("ambiguous")).join(ns1, on="country")
# ~97.7% of empty-address queries are true matches (train); loss ~ k_pool * ambiguous_true / n_s1
r = r.with_columns((0.977 * pl.col("ambiguous") / pl.col("n_s1")).alias("amb_per_s1")).with_columns((pl.col("amb_per_s1") * k_pool).round(5).alias("est_irreducible_F_loss"))
w = r.with_columns((pl.col("n_s1") / pl.col("n_s1").sum()).alias("w"))
print(w.sort("country"))
print("LB-weighted irreducible loss:", round((w["w"] * w["est_irreducible_F_loss"]).sum(), 5), "-> LB ceiling ~", round(1 - (w["w"] * w["est_irreducible_F_loss"]).sum(), 4))
