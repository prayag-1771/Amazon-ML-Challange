import sys; sys.path.insert(0, ".")
import polars as pl
W = "../work/"
sc = pl.read_parquet(W + "test_scores_stage2_v11.parquet").join(pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id")
top = sc.sort("p2", descending=True).unique("s23_id", keep="first").filter(pl.col("country") == "France")
m = pl.read_csv(W + "probe11_fr_vfix/matching_results.tsv", separator="\t", schema_overrides={"matched_entity_ids": pl.String}).rename({"source1_entity_id": "s1_id"})
acc = m.filter(pl.col("matched_entity_ids").fill_null("") != "").with_columns(pl.col("matched_entity_ids").str.split(",").alias("s23_id")).explode("s23_id").select("s1_id", "s23_id", pl.lit(True).alias("acc"))
nacc = m.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len().alias("nacc")).select("s1_id", "nacc")
top = top.join(acc, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("acc").fill_null(False))
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "legal", "name_core", "nums"]).rename({"entity_id": "s1_id", "legal": "l1", "name_core": "n1", "nums": "u1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id", "legal", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "legal": "lq", "name_core": "nq", "nums": "uq"})
d = top.filter((pl.col("p") >= .9) & ~pl.col("acc")).join(s1, on="s1_id").join(q, on="s23_id")
l1, lq = pl.col("l1").fill_null(""), pl.col("lq").fill_null("")
a = pl.col("u1").str.split(" ").list.first(); b = pl.col("uq").str.split(" ").list.first()
d = d.with_columns(pl.when(l1 == lq).then(pl.lit("same")).when(l1 == "").then(pl.lit("add")).when(lq == "").then(pl.lit("drop")).otherwise(pl.lit("chg")).alias("leg"),
                   pl.when(pl.col("n1") == pl.col("nq")).then(pl.lit("name=")).otherwise(pl.lit("name~")).alias("nm"),
                   pl.when(a.is_null() | b.is_null()).then(pl.lit("miss")).when(a == b).then(pl.lit("num=")).otherwise(pl.lit("num~")).alias("nu"))
print("France top-1 p>=.9 not accepted by vfix:", d.height)
print(d.group_by("nm", "nu", "leg").agg(pl.len(), pl.col("p2").median().round(3)).sort("len", descending=True).head(20))
d.write_parquet(W + "v10/down_fr.parquet")
