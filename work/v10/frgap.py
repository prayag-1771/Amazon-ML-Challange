"""Top-1 pairs by category: US/India valid true rate + acceptance vs France test acceptance (v12), per 1000 S1."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
W = "../work/"
T = {"US": .75, "India": .8}
def cat(split, sc):
    s1 = pl.read_parquet(W+f"norm_{split}_s1.parquet", columns=["entity_id","country","name_core","nums","legal","addr_empty"]).rename({"entity_id":"s1_id","name_core":"n1","nums":"u1","legal":"l1","addr_empty":"e1"})
    q = pl.concat([pl.read_parquet(W+f"norm_{split}_s{i}.parquet", columns=["entity_id","name_core","nums","legal","addr_empty"]) for i in (2,3)]).rename({"entity_id":"s23_id","name_core":"nq","nums":"uq","legal":"lq","addr_empty":"eq"})
    x = sc.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
    f1 = pl.col("u1").fill_null("").str.split(" ").list.first(); fq = pl.col("uq").fill_null("").str.split(" ").list.first()
    t1 = pl.col("n1").str.split(" "); tq = pl.col("nq").str.split(" ")
    x = x.with_columns(
        pl.when(pl.col("eq") == 1).then(pl.lit("qempty")).when((f1 == "") | (fq == "")).then(pl.lit("miss")).when(f1 == fq).then(pl.lit("num=")).otherwise(pl.lit("num~")).alias("nu"),
        pl.when(pl.col("n1") == pl.col("nq")).then(pl.lit("name=")).otherwise(
            pl.format("+{}-{}", tq.list.set_difference(t1).list.len().clip(upper_bound=2), t1.list.set_difference(tq).list.len().clip(upper_bound=2))).alias("nm"),
        pl.when(pl.col("l1") == pl.col("lq")).then(pl.lit("same")).when(pl.col("l1") == "").then(pl.lit("add")).when(pl.col("lq") == "").then(pl.lit("drop")).otherwise(pl.lit("chg")).alias("leg"))
    return x
v = cat("train", pl.read_parquet(W+"valid_scores_stage2.parquet")).filter(is_valid_expr("s1_id"))
v = v.with_columns((pl.col("p2") >= pl.col("country").replace_strict(T)).alias("acc"))
m = pl.read_parquet(W+"v10/v12_matches.parquet").select("s1_id","s23_id", pl.lit(True).alias("acc"))
t = cat("test", pl.read_parquet(W+"test_scores_stage2_v12.parquet")).filter(pl.col("country") == "France").join(m, on=["s1_id","s23_id"], how="left").with_columns(pl.col("acc").fill_null(False))
nv = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).group_by("country").len("ns1")
nf = (pl.read_parquet(W+"norm_test_s1.parquet", columns=["country"])["country"] == "France").sum()
K = ["nu","nm","leg"]
a = v.join(nv, on="country").group_by(K).agg((pl.len()/pl.col("ns1").first()*1000/2).alias("ui_perK"), pl.col("label").mean().alias("ui_true"), pl.col("acc").mean().alias("ui_acc"))
b = t.group_by(K).agg((pl.len()/nf*1000).alias("fr_perK"), pl.col("acc").mean().alias("fr_acc"), pl.col("p").mean().alias("fr_p1"))
r = b.join(a, on=K, how="left").with_columns(
    (pl.col("fr_perK")*pl.col("fr_acc")).alias("fr_accK"),
    (pl.col("fr_perK")*(pl.col("ui_true") - pl.col("fr_acc"))).alias("gapK"))
print(r.sort("fr_perK", descending=True).head(40).with_columns(pl.selectors.float().round(3)))
print("sorted by |gapK|"); print(r.filter(pl.col("fr_perK") > 5).sort(pl.col("gapK").abs(), descending=True).head(20).with_columns(pl.selectors.float().round(3)))
t.write_parquet(W+"v10/fr_cat_v12.parquet"); v.write_parquet(W+"v10/ui_cat_v12.parquet")
