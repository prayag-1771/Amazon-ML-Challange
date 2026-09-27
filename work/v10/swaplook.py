import sys
sys.path.insert(0, ".")
import polars as pl
W = "../work/"
exec(open("../work/v10/frcat.py").read().split("def load")[0].split("W = \"../work/\"")[1])
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country", "name_core", "nums", "business_name", "business_address"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id", "name_core", "nums", "business_name", "business_address"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq", "business_name": "qn", "business_address": "qa"})
d = pl.read_parquet(W + "test_scores_stage2_v9.parquet").join(s1.select("s1_id", "country"), on="s1_id").filter(pl.col("country") == "France")
top = cats(d.sort("p2", descending=True).unique("s23_id", keep="first").join(s1.drop("country"), on="s1_id").join(q, on="s23_id"))
t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
x = top.filter((pl.col("nm") == "swap1") & (pl.col("num") == "num=")).with_columns(tq.list.set_difference(t1).list.first().alias("at"))
for tok in ("developpement", "groupe", "sportive", "fils"):
    ids = x.filter(pl.col("at") == tok).sample(4, seed=3)["s1_id"]
    for sid in ids:
        r = s1.filter(pl.col("s1_id") == sid).row(0, named=True)
        print(f"\n#### [{tok}] S1: {r['business_name']} | {r['business_address']}")
        c = d.filter(pl.col("s1_id") == sid).join(q, on="s23_id").join(top.select("s23_id", pl.col("s1_id").alias("tops1"), pl.col("p2").alias("qmax")), on="s23_id").sort("p2", descending=True)
        for rr in c.head(12).iter_rows(named=True):
            print(f"   p2={rr['p2']:.3f} {'   ' if rr['tops1']==sid else 'ALT'} {rr['s23_id'][:2]} {rr['qn']} | {rr['qa']}")
