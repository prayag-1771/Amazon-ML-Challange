import sys; sys.path.insert(0, ".")
import polars as pl
def pairs(f):
    m = pl.read_csv(f, separator="\t")
    return m.select(pl.col(m.columns[0]).alias("s1_id"), pl.col(m.columns[1]).str.split(",").alias("s23_id")).explode("s23_id").drop_nulls()
c = pl.read_parquet("../work/norm_test_s1.parquet", columns=["entity_id", "country", "business_name", "business_address", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
q = pl.concat([pl.read_parquet(f"../work/norm_test_s{i}.parquet", columns=["entity_id", "business_name", "business_address", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "business_name": "qn", "business_address": "qa", "name_core": "nq", "nums": "uq"})
a = pairs("../work/v11_noceh_matching.tsv"); b = pairs("../output/matching_results.tsv")
fr = c.filter(pl.col("country") == "France")
print("France S1", len(fr), "acc/S1 noceh", round(a.join(fr, on="s1_id").height / len(fr), 3), "ceh", round(b.join(fr, on="s1_id").height / len(fr), 3))
lost = a.join(b, on=["s1_id", "s23_id"], how="anti").join(fr, on="s1_id").join(q, on="s23_id")
gain = b.join(a, on=["s1_id", "s23_id"], how="anti").join(fr, on="s1_id").join(q, on="s23_id")
print("lost", len(lost), "gained", len(gain))
f1 = pl.col("u1").str.split(" ").list.first(); fq = pl.col("uq").str.split(" ").list.first()
t1 = pl.col("n1").fill_null("").str.split(" "); tq = pl.col("nq").fill_null("").str.split(" ")
for nm, x in (("lost", lost), ("gain", gain)):
    x = x.with_columns((f1 == fq).alias("num_eq"), tq.list.set_difference(t1).list.len().alias("add"), t1.list.set_difference(tq).list.len().alias("drop"))
    print(nm, x.group_by("num_eq", "add", "drop").len().sort("len", descending=True).head(8).rows())
    for r in x.sample(12, seed=1).select("business_name", "business_address", "qn", "qa").rows(): print("   ", r)
