"""Does keying the label-free numeq token stat by (tok, kind, pos) track truth better? Train (labels) + France test view."""
import sys
sys.path.insert(0, ".")
import numpy as np
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
W = "../work/"
M = 20
gt = pl.read_parquet(W + "gt_pairs.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
def pairs(split):
    c = pl.read_parquet(W + f"pruned_{split}.parquet", columns=["q_row", "s1_row", "p0_qrank"]).filter(pl.col("p0_qrank") == 1)
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_full", "nums"])
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{k}.parquet", columns=["entity_id", "name_full", "nums"]) for k in (2, 3)])
    A = s1[c["s1_row"].to_numpy()].rename({"entity_id": "s1_id"})
    B = q[c["q_row"].to_numpy()].rename({"entity_id": "s23_id", "name_full": "nb", "nums": "mb"})
    d = pl.concat([A, B], how="horizontal")
    ta = pl.col("name_full").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    tb = pl.col("nb").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    d = d.with_columns(ta.alias("ta"), tb.alias("tb")).with_columns(
        pl.col("tb").list.set_difference(pl.col("ta")).alias("xq"), pl.col("ta").list.set_difference(pl.col("tb")).alias("x1"),
        pl.when((pl.col("nums") == "") | (pl.col("mb") == "")).then(None).otherwise(
            pl.col("mb").str.split(" ").list.contains(pl.col("nums").str.split(" ").list.first()).cast(pl.Float32)).alias("numeq"))
    d = d.with_columns(pl.when(pl.col("x1").list.len() == 0).then(pl.lit("a")).otherwise(pl.lit("s")).alias("kq"),
                       pl.when(pl.col("xq").list.len() == 0).then(pl.lit("a")).otherwise(pl.lit("s")).alias("k1"))
    if split == "train":
        d = d.join(gt, on=["s1_id", "s23_id"], how="left").with_columns(pl.col("y").fill_null(0))
    return d
def explode(d, side, kcol, tcol):
    e = d.select("country", "numeq", *(["y"] if "y" in d.columns else []), pl.col(kcol).alias("kind"), pl.col(tcol).alias("toks"), pl.col(side).alias("tok")).explode("tok").drop_nulls("tok")
    n = pl.col("toks").list.len()
    idx = pl.col("toks").list.eval(pl.element()).list.len()  # placeholder
    return e.with_columns(pl.when(pl.col("toks").list.first() == pl.col("tok")).then(pl.lit("P"))
                          .when(pl.col("toks").list.last() == pl.col("tok")).then(pl.lit("S")).otherwise(pl.lit("M")).alias("pos")).drop("toks")
def rates(e, keys):
    prior = e.group_by("country").agg(pl.col("numeq").mean().alias("prior"))
    g = e.group_by(["country"] + keys).agg(pl.col("numeq").sum().alias("s"), pl.col("numeq").count().alias("k"))
    g = g.join(prior, on="country").with_columns(((pl.col("s") + M * pl.col("prior")) / (pl.col("k") + M) / pl.col("prior")).alias("rel"))
    return g.select(["country"] + keys + ["rel"])
tr = pairs("train")
for side, kc, tc in (("xq", "kq", "tb"), ("x1", "k1", "ta")):
    e = explode(tr, side, kc, tc)
    for keys in (["tok"], ["tok", "kind"], ["tok", "kind", "pos"]):
        r = rates(e, keys)
        z = e.join(r, on=["country"] + keys)
        # per-occurrence: how well rel separates truth -> AUC and calibration correlation per key group
        from sklearn.metrics import roc_auc_score
        ys = z["y"].to_numpy(); rs = z["rel"].to_numpy()
        grp = z.group_by(["country"] + keys).agg(pl.len(), pl.col("y").mean().alias("t"), pl.col("rel").first())
        w = grp["len"].to_numpy(); cc = np.corrcoef(grp["t"].to_numpy(), grp["rel"].to_numpy())
        print(side, keys, "AUC", round(roc_auc_score(ys, rs), 4), "groups", grp.height, flush=True)
te = pairs("test")
e = explode(te, "xq", "kq", "tb")
r = rates(e, ["tok", "kind", "pos"])
print(r.filter((pl.col("country") == "France") & pl.col("tok").is_in(["groupe", "developpement", "club", "sportive", "france", "holding", "sas", "fils", "ecole", "amicale", "comite", "union"])).sort("tok", "kind", "pos"))
r.write_parquet(W + "tokkey_test_xq.parquet")
