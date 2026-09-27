import polars as pl
W = "../work/"
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
def asg(name, t):
    d = pl.read_parquet(W + f"test_scores_{name}.parquet")
    d = d.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") >= t)
    return d.join(s1, on="s1_id")
a = asg("lgb_v5cf", 0.75); b = asg("lgb_v6k", 0.70)
for c in ["US", "India", "France"]:
    x = a.filter(pl.col("country") == c).select("s1_id", "s23_id"); y = b.filter(pl.col("country") == c).select("s1_id", "s23_id")
    both = x.join(y, on=["s1_id", "s23_id"]).height
    print(c, "v5cf", x.height, "v6k", y.height, "only_v5cf", x.height - both, "only_v6k", y.height - both)
# France changes, token context
n1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "name_full", "addr_core"]).rename({"entity_id": "s1_id", "name_full": "n1", "addr_core": "a1"})
q = pl.concat([pl.read_parquet(W + f"norm_test_s{k}.parquet", columns=["entity_id", "name_full", "addr_core"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "name_full": "nq", "addr_core": "aq"})
x = a.filter(pl.col("country") == "France"); y = b.filter(pl.col("country") == "France")
for lab, u, v in (("ONLY v6k", y, x), ("ONLY v5cf", x, y)):
    d = u.join(v, on=["s1_id", "s23_id"], how="anti").join(n1, on="s1_id").join(q, on="s23_id").sample(12, seed=1)
    print("==", lab)
    for r in d.iter_rows(named=True):
        print(f"  p={r['p']:.3f} {r['n1']} | {r['a1']}\n          {r['nq']} | {r['aq']}")
