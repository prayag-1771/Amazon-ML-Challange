import polars as pl
W = "../work/"
t = pl.read_parquet(W + "v10/fr_top1.parquet")
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "business_name", "business_address"]).rename({"entity_id": "s1_id", "business_name": "b1", "business_address": "ad1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{i}.parquet", columns=["entity_id", "business_name", "business_address"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "business_name": "bq", "business_address": "adq"})
t = t.join(s1, on="s1_id").join(q, on="s23_id")
def show(nm, x, n=12):
    print(f"\n===== {nm}: {x.height}")
    for r in x.sample(min(n, x.height), seed=5).sort("p2").iter_rows(named=True):
        print(f"  p2={r['p2']:.3f} acc={int(r['acc'])} | {r['b1']} | {r['ad1']}\n{'':22}| {r['bq']} | {r['adq']}")
for e in ["far", "near-", "drop", "missing"]:
    show(f"name= {e} REJ p2>.02", t.filter((pl.col("k") == "name=") & (pl.col("e") == e) & ~pl.col("acc") & (pl.col("p2") > 0.02)))
    show(f"name= {e} ACC", t.filter((pl.col("k") == "name=") & (pl.col("e") == e) & pl.col("acc")), 6)
show("swap1_typo eq REJ", t.filter((pl.col("k") == "swap1_typo") & (pl.col("e") == "eq") & ~pl.col("acc")))
show("swap1_var eq REJ", t.filter((pl.col("k") == "swap1_var") & (pl.col("e") == "eq") & ~pl.col("acc")))
