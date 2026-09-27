import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
W = "../work/"
T = {"US": 0.75, "India": 0.80, "France": 0.75}
D3 = [3, 4, 5, 7, 9, 11, 13, 21]
def prep(sc, split, filt=None):
    c = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "legal", "name_core", "nums"]).rename({"entity_id": "s1_id", "legal": "l1", "name_core": "n1", "nums": "u1"})
    if filt is not None: c = c.filter(filt)
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "legal", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "legal": "lq", "name_core": "nq", "nums": "uq"})
    d = pl.read_parquet(W + sc)
    top = d.sort("p2", descending=True).unique("s23_id", keep="first").filter(pl.col("p2") > 0.02).join(c, on="s1_id").join(q, on="s23_id")
    a = pl.col("u1").str.split(" ").list.first().cast(pl.Int64, strict=False); b = pl.col("uq").str.split(" ").list.first().cast(pl.Int64, strict=False)
    top = top.with_columns((b - a).alias("dif"),
        pl.when(pl.col("l1").fill_null("") == pl.col("lq").fill_null("")).then(pl.lit("same")).when(pl.col("l1").fill_null("") == "").then(pl.lit("add")).when(pl.col("lq").fill_null("") == "").then(pl.lit("drop")).otherwise(pl.lit("chg")).alias("leg"),
        (pl.col("n1") == pl.col("nq")).alias("nsame"), (pl.col("p2") >= pl.col("country").replace_strict(T)).alias("acc"))
    ad = pl.col("dif").abs()
    top = top.with_columns(pl.when(ad == 1).then(pl.lit("d1")).when(ad == 2).then(pl.lit("d2")).when(ad.is_in(D3)).then(pl.lit("d3")).when(ad.is_between(1, 25)).then(pl.lit("dx")).otherwise(None).alias("dg"),
                           (pl.col("dif") > 0).alias("up"))
    return top.filter(pl.col("dg").is_not_null())
v = prep("valid_scores_stage2.parquet", "train", is_valid_expr("s1_id"))
t = prep("test_scores_stage2_v10.parquet", "test")
v.write_parquet(W + "shift_v.parquet"); t.write_parquet(W + "shift_t.parquet")
K = ["country", "dg", "leg", "nsame"]
def cell(x, lab):
    ag = [pl.len().alias("n"), pl.col("acc").sum().alias("a")] + ([pl.col("label").sum().alias("pos"), (pl.col("acc") & (pl.col("label") == 1)).sum().alias("tp")] if lab else [])
    g = x.group_by(K + ["up"]).agg(ag)
    return g.filter(pl.col("up")).drop("up").join(g.filter(~pl.col("up")).drop("up"), on=K, suffix="_dn", how="full", coalesce=True).fill_null(0)
cv, ct = cell(v, True), cell(t, False)
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250)
print("VAL pooled per country/dg:", cv.group_by("country", "dg").agg(pl.col("n", "pos", "a", "tp", "n_dn", "pos_dn").sum()).sort("country", "dg").rows())
x = ct.join(cv, on=K, how="left", suffix="_v").fill_null(0)
x = x.with_columns((pl.col("n_dn") * 0.95).alias("true_up_est")).with_columns((pl.col("a") - pl.col("true_up_est").clip(upper_bound=pl.col("a"))).round(0).alias("fp_est"))
print(x.group_by("country").agg(pl.col("fp_est").sum(), pl.col("a").sum()).rows())
print(x.filter(pl.col("n") > 1500).sort("fp_est", descending=True).select(K + ["n", "a", "n_dn", "a_dn", "fp_est", "n_v", "pos_v", "a_v", "tp_v", "n_dn_v"]).head(40))
