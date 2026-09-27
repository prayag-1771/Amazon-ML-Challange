"""Sibling confusion: S1 entities sharing an address, and assigned queries whose 2nd-best S1 is also plausible."""
import polars as pl

W = "../work/"
s1 = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country", "addr_full", "name_full"])
tr1 = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country", "addr_full", "name_full"])
for nm, d in (("train", tr1), ("test", s1)):
    d = d.filter(pl.col("addr_full") != "")
    d = d.with_columns(pl.len().over("country", "addr_full").alias("k"))
    print(nm, d.group_by("country").agg((pl.col("k") > 1).mean().alias("s1_sharing_addr"),
                                        pl.col("k").mean().alias("mean_k")).sort("country"))

sc = pl.read_parquet(W + "test_scores_lgb_v4a.parquet").join(
    s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id")
r = sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("s23_id").alias("r"))
top = r.filter(pl.col("r") == 1).select("s23_id", "country", pl.col("p").alias("p1"), pl.col("s1_id").alias("s1a"))
sec = r.filter(pl.col("r") == 2).select("s23_id", pl.col("p").alias("p2"), pl.col("s1_id").alias("s1b"))
t = top.join(sec, on="s23_id", how="left").with_columns(pl.col("p2").fill_null(0))
a = t.filter(pl.col("p1") >= 0.75)
print(a.group_by("country").agg(pl.len(), (pl.col("p2") > 0.05).mean().alias("p2>0.05"),
                                (pl.col("p2") > 0.2).mean().alias("p2>0.2"),
                                ((pl.col("p1") + pl.col("p2")) > 1.0).mean().alias("sum>1")).sort("country"))
u = t.filter(pl.col("p1") < 0.75)
print("unassigned:", u.group_by("country").agg(pl.len(), (pl.col("p1") > 0.3).mean().alias("p1>0.3"),
                                               ((pl.col("p1") > 0.3) & (pl.col("p2") > 0.3)).mean().alias("both>0.3")).sort("country"))
fr = a.filter((pl.col("country") == "France") & (pl.col("p2") > 0.2)).sample(20, seed=0)
nm = s1.select("entity_id", "name_full", "addr_full")
q = pl.concat([pl.read_parquet(W + f"norm_test_s{k}.parquet", columns=["entity_id", "name_full", "addr_full"]) for k in (2, 3)])
fr = (fr.join(q.rename({"entity_id": "s23_id", "name_full": "q_name", "addr_full": "q_addr"}), on="s23_id")
        .join(nm.rename({"entity_id": "s1a", "name_full": "a_name", "addr_full": "a_addr"}), on="s1a")
        .join(nm.rename({"entity_id": "s1b", "name_full": "b_name", "addr_full": "b_addr"}), on="s1b"))
pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(400)
print(fr.select("p1", "p2", "q_name", "a_name", "b_name", "q_addr", "a_addr", "b_addr"))
