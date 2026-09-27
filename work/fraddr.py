"""Full picture at French addresses shared by several S1: all S1, all queries (exact addr), current assignment."""
import sys; sys.path.insert(0, ".")
import polars as pl
W="../work/"
s1 = pl.read_parquet(W+"norm_test_s1.parquet", columns=["entity_id","country","business_name","business_address","addr_core"]).rename({"entity_id":"s1_id"})
q = pl.concat([pl.read_parquet(W+f"norm_test_s{k}.parquet", columns=["entity_id","country","business_name","business_address","addr_core"]).with_columns(pl.lit(k).alias("src")) for k in (2,3)]).rename({"entity_id":"s23_id"})
sc = pl.read_parquet(W+"test_scores_lgb_v6k.parquet")
best = sc.sort("p", descending=True).unique("s23_id", keep="first").rename({"s1_id":"best","p":"pb"})
f1 = s1.filter((pl.col("country")=="France") & (pl.col("addr_core").str.len_chars()>5)).with_columns(pl.len().over("addr_core").alias("n"))
import random
random.seed(3)
for n in (2, 3, 4):
    addrs = f1.filter(pl.col("n")==n)["addr_core"].unique().sort().to_list()
    for a in random.sample(addrs, 4):
        ss = f1.filter(pl.col("addr_core")==a)
        lab = {r: chr(65+i) for i, r in enumerate(ss["s1_id"].to_list())}
        print(f"\n### {a}  (n_s1={n})")
        for r in ss.iter_rows(named=True):
            print(f"  S1 {lab[r['s1_id']]}: {r['business_name']} | {r['business_address']}")
        qq = q.filter((pl.col("country")=="France") & (pl.col("addr_core")==a)).join(best, on="s23_id", how="left")
        # also queries assigned to these S1 with a different addr
        qa = q.join(best, on="s23_id").filter(pl.col("best").is_in(ss["s1_id"].implode()) & (pl.col("addr_core")!=a))
        for r in pl.concat([qq, qa]).sort("business_name").iter_rows(named=True):
            print(f"    s{r['src']} {r['business_name']} | {r['business_address']}  -> {lab.get(r['best'], '?other')} p={r['pb'] if r['pb'] is None else round(r['pb'],3)}")
