import sys; sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250)
W = "../work/"
exec(open("../work/cells.py", encoding="utf-8").read().split("va = load")[0].split("pl.Config")[0])
exec("def load" + open("../work/cells.py", encoding="utf-8").read().split("def load")[1].split("va = load")[0])
te = load("test", "test_scores_lgb_v6k.parquet")
raw1 = pl.read_parquet(W + "raw_test_s1.parquet", columns=["entity_id", "business_name", "business_address"]).rename({"entity_id": "s1_id", "business_name": "N1", "business_address": "A1"})
rawq = pl.concat([pl.read_parquet(W + f"raw_test_s{k}.parquet", columns=["entity_id", "business_name", "business_address"]) for k in (2, 3)]).rename({"entity_id": "s23_id", "business_name": "NQ", "business_address": "AQ"})
for nm, ad in (("eq", "numdiff"), ("swap", "same"), ("added", "empty")):
    d = te.filter((pl.col("country") == "France") & (pl.col("name") == nm) & (pl.col("addr") == ad)).join(raw1, on="s1_id").join(rawq, on="s23_id")
    print("=====", nm, ad, d.height, "p quantiles", [round(d["p"].quantile(x), 3) for x in (0.1, 0.25, 0.5, 0.75, 0.9)])
    for r in d.sample(14, seed=5).iter_rows(named=True):
        print(f"  p={r['p']:.3f} | {r['N1']} | {r['A1']}\n            | {r['NQ']} | {r['AQ']}   [m1={r['m1']} mq={r['mq']}]")
# US for reference
d = te.filter((pl.col("country") == "US") & (pl.col("name") == "eq") & (pl.col("addr") == "numdiff")).join(raw1, on="s1_id").join(rawq, on="s23_id")
print("===== US eq numdiff")
for r in d.sample(8, seed=5).iter_rows(named=True):
    print(f"  p={r['p']:.3f} | {r['N1']} | {r['A1']}\n            | {r['NQ']} | {r['AQ']}   [m1={r['m1']} mq={r['mq']}]")
