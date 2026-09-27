import polars as pl, random
W = "../work/"
t = pl.read_parquet(W + "v10/fr_top1.parquet")
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "business_name", "business_address"]).rename({"entity_id": "s1_id", "business_name": "b1", "business_address": "ad1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id", "business_name", "business_address"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "business_name": "bq", "business_address": "adq"})
t = t.join(s1, on="s1_id").join(q, on="s23_id")
def show(nm, x, n=14):
    print(f"\n===== {nm}: {x.height}")
    for r in x.sample(min(n, x.height), seed=3).sort("p2").iter_rows(named=True):
        print(f"  p2={r['p2']:.3f} acc={int(r['acc'])} | {r['b1']} | {r['ad1']}\n{'':22}| {r['bq']} | {r['adq']}")
show("name= eq REJECTED", t.filter((pl.col("k") == "name=") & (pl.col("e") == "eq") & ~pl.col("acc")))
x = t.filter((pl.col("k") == "addonly_word") & (pl.col("e") == "eq"))
print(x.group_by("at").agg(pl.len(), pl.col("acc").mean().round(2), pl.col("p2").median().round(3)).sort("len", descending=True).head(25))
show("addonly_word eq ACCEPTED", x.filter(pl.col("acc")))
x = t.filter((pl.col("k") == "swap1_word") & (pl.col("e") == "eq"))
print(x.group_by("at").agg(pl.len(), pl.col("acc").mean().round(2), pl.col("p2").median().round(3)).sort("len", descending=True).head(25))
show("swap1_word eq ACCEPTED", x.filter(pl.col("acc")))
show("other eq REJECTED", t.filter((pl.col("k") == "other") & (pl.col("e") == "eq") & ~pl.col("acc") & (pl.col("p2") > 0.05)))
show("other eq ACCEPTED", t.filter((pl.col("k") == "other") & (pl.col("e") == "eq") & pl.col("acc")))
