import sys; sys.path.insert(0, ".")
import polars as pl
W="../work/"
miss=pl.read_parquet(W+"miss_v12.parquet").filter(~pl.col("qe"))
s1=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","nums","addr_core"]).rename({"entity_id":"s1_id","nums":"u1f","addr_core":"a1f"})
q=pl.concat([pl.read_parquet(W+f"norm_train_s{i}.parquet",columns=["entity_id","nums","addr_core","alpha_comps"]) for i in (2,3)]).rename({"entity_id":"s23_id"})
m=miss.join(s1,on="s1_id").join(q,on="s23_id")
num_in = pl.col("nums").str.split(" ").list.set_intersection(pl.col("u1f").str.split(" ")).list.len()>0
lastc = pl.col("alpha_comps").fill_null("").str.split("|").list.last()
m=m.with_columns(num_in.alias("num_in"), lastc.alias("lc"))
m=m.with_columns(pl.struct("lc","a1f").map_elements(lambda r: bool(r["lc"]) and r["lc"] in (r["a1f"] or ""),return_dtype=pl.Boolean).alias("city_in"))
print(m.group_by("country","num_in","city_in").len().sort("country","len",descending=[False,True]).rows())
