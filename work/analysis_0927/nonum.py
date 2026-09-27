"""Rule candidate: query has no house number, top-1 S1 has one; name_core equal (or equal up to one variant/legal
token) and street words overlap. Precision + F gain on US/India validation, flip count on France test."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "business_entity_resolution"))
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(220)
W = str(__import__("pathlib").Path(__file__).resolve().parents[1]) + "/"
T = {"US": 0.75, "India": 0.80, "France": 0.75}
COLS = ["entity_id", "name_core", "nums", "addr_core", "alpha_comps"]


def side(split, top):
    ids1 = top.select("s1_id").unique().lazy()
    idq = top.select("s23_id").unique().lazy()
    s1 = (pl.scan_parquet(W + f"norm_{split}_s1.parquet").select(COLS + ["country"]).rename(
        {"entity_id": "s1_id", "name_core": "n1", "nums": "u1", "addr_core": "a1", "alpha_comps": "c1"})
        .join(ids1, on="s1_id", how="semi").collect())
    q = (pl.concat([pl.scan_parquet(W + f"norm_{split}_s{k}.parquet").select(COLS + ["addr_empty"]) for k in (2, 3)]).rename(
        {"entity_id": "s23_id", "name_core": "nq", "nums": "uq", "addr_core": "aq", "alpha_comps": "cq"})
        .join(idq, on="s23_id", how="semi").collect())
    return s1, q


def tag(top, split):
    s1, q = side(split, top)
    x = top.join(s1.drop("country"), on="s1_id").join(q, on="s23_id")
    tok = lambda a, c: pl.col(a).fill_null("").str.split(" ").list.set_difference(
        pl.col(c).fill_null("").str.replace_all(r"\|", " ").str.split(" ")).list.eval(pl.element().filter((pl.element() != "") & ~pl.element().str.contains(r"\d")))
    x = x.with_columns(tok("a1", "c1").alias("t1"), tok("aq", "c1").alias("tq"),
                       pl.col("n1").str.split(" ").alias("w1"), pl.col("nq").str.split(" ").alias("wq"))
    x = x.with_columns((pl.col("t1").list.set_intersection("tq").list.len() / pl.max_horizontal(pl.col("t1").list.len(), pl.col("tq").list.len(), 1)).alias("street_ov"),
                       pl.col("wq").list.set_difference("w1").list.len().alias("na"), pl.col("w1").list.set_difference("wq").list.len().alias("nd"))
    nonum = (pl.col("uq") == "") & (pl.col("u1") != "") & (pl.col("addr_empty") == 0)
    return x.with_columns(
        (nonum & (pl.col("n1") == pl.col("nq")) & (pl.col("street_ov") >= 0.5)).alias("r_eq"),
        (nonum & (pl.col("na") <= 1) & (pl.col("nd") == 0) & (pl.col("street_ov") >= 0.5)).alias("r_add1"),
    )


# ---------------- validation (US/India)
gt = pl.read_parquet(W + "gt_pairs.parquet")
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}).filter(is_valid_expr("s1_id"))
gtv = gt.join(s1v, on="s1_id", how="semi")
pred = pl.read_parquet(W + "pred15_v.parquet").select("s1_id", "s23_id")
v = pl.read_parquet(W + "valid_scores_stage2.parquet").join(s1v, on="s1_id")
vtop = tag(v.sort("p2", descending=True).unique("s23_id", keep="first"), "train").with_columns(pl.col("country").replace_strict(T).alias("th"))
del v
base = macro_f05(s1v["s1_id"], pred, gtv, by=s1v)
for r in ("r_eq", "r_add1"):
    sel = vtop.filter(pl.col(r))
    flip = sel.filter(pl.col("p2") < pl.col("th")).join(pred, on="s23_id", how="anti")
    for pmin in (0.0, 0.01, 0.1):
        f = flip.filter(pl.col("p2") >= pmin)
        mm = macro_f05(s1v["s1_id"], pl.concat([pred, f.select("s1_id", "s23_id")]), gtv, by=s1v)
        print(f"VAL {r} pmin={pmin}: cell {sel.height} (label rate {(sel['label'].mean() or 0):.4f}), flips {f.height} precision {(f['label'].mean() or 0) if f.height else float('nan'):.3f}, "
              f"F {mm['f05'] - base['f05']:+.5f} (US {mm['f05_US'] - base['f05_US']:+.5f} IN {mm['f05_India'] - base['f05_India']:+.5f})")
# ---------------- test France
del vtop, gt, gtv, pred
t = pl.read_parquet(W + "test_scores_stage2_v15.parquet").join(pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"}), on="s1_id")
ttop = tag(t.filter(pl.col("country") == "France").sort("p2", descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id", "p2", "country"), "test").with_columns(pl.col("country").replace_strict(T).alias("th"))
for r in ("r_eq", "r_add1"):
    print("TEST", r, ttop.filter(pl.col(r)).group_by("country").agg(pl.len().alias("cell"), (pl.col("p2") < pl.col("th")).sum().alias("rejected_now"),
          ((pl.col("p2") < pl.col("th")) & (pl.col("p2") >= 0.01)).sum().alias("rejected_p2>=0.01")).sort("country").rows())
