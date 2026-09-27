import polars as pl
W = "../work/"
t = pl.read_parquet("../work/v10/fr_nameeq.parquet")
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "business_name", "business_address", "legal"]).rename({"entity_id": "s1_id", "business_name": "b1", "business_address": "ad1", "legal": "l1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id", "business_name", "business_address", "legal"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "business_name": "bq", "business_address": "adq", "legal": "lq"})
x = t.join(s1, on="s1_id").join(q, on="s23_id").filter((pl.col("ov") > 0.34) & ~pl.col("acc") & (pl.col("p2") > 0.1) & (pl.col("l1").fill_null("") == "") & (pl.col("lq").fill_null("") != ""))
sc = pl.read_parquet(W + "test_scores_stage2_v11.parquet")
m = pl.read_csv(W + "probe11_fr_vfix/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).rename({"source1_entity_id": "s1_id"})
m = m.filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(",").alias("s23_id")).explode("s23_id").select("s1_id", "s23_id", pl.lit(1).alias("A"))
for r in x.sample(10, seed=2).iter_rows(named=True):
    print(f"\n#### Q {r['bq']} | {r['adq']}  (top1 p2={r['p2']:.3f})")
    for c in sc.filter(pl.col("s23_id") == r["s23_id"]).sort("p2", descending=True).head(4).join(s1, on="s1_id").iter_rows(named=True):
        print(f"    S1 p={c['p']:.3f} p2={c['p2']:.3f} {c['b1']} | {c['ad1']}")
    print(f"  -- other queries of S1 {r['b1']}:")
    for c in sc.filter(pl.col("s1_id") == r["s1_id"]).sort("p2", descending=True).head(8).join(q, on="s23_id").join(m, on=["s1_id", "s23_id"], how="left").iter_rows(named=True):
        print(f"    acc={c['A'] or 0} p2={c['p2']:.3f} {c['bq']} | {c['adq']}")
