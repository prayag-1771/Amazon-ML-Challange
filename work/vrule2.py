"""Data-driven variant-vocab rule (label-free, all countries). Compare with probe11_fr_vfix on test, and measure on val."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "../work")
import polars as pl
from rapidfuzz.distance import Levenshtein as L
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W = "../work/"; T = {"US": 0.75, "India": 0.80, "France": 0.75}
IN_MIN, IO_MIN, RW_MIN = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
def tops(split, sc):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    if split == "train": s1 = s1.filter(is_valid_expr("s1_id"))
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    sp = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    a = pl.col("u1").str.split(" ").list.first(); b = pl.col("uq").str.split(" ").list.first()
    def prep(t):
        t = t.join(s1, on="s1_id").join(q, on="s23_id").with_columns(sp("n1").alias("t1"), sp("nq").alias("tq"), (a == b).fill_null(False).alias("eq"))
        t = t.with_columns(pl.col("tq").list.set_difference("t1").alias("A"), pl.col("t1").list.set_difference("tq").alias("D"))
        return t.with_columns(pl.col("A").list.len().alias("na"), pl.col("D").list.len().alias("nd"), pl.col("A").list.first().alias("at"), pl.col("D").list.first().alias("dt"))
    t0 = prep(sc.sort("p", descending=True).unique("s23_id", keep="first"))  # stage-1 top-1: label-free token stats
    ns = s1.group_by("country").len("ns")
    sw = t0.filter((pl.col("na") == 1) & (pl.col("nd") == 1) & pl.col("eq"))
    st = s1.select("country", sp("n1").alias("tok")).explode("tok").drop_nulls().group_by("country", "tok").len("s1f")
    for z in (sw.group_by("country", pl.col("at").alias("tok")).len("sin"), sw.group_by("country", pl.col("dt").alias("tok")).len("sout")):
        st = st.join(z, on=["country", "tok"], how="full", coalesce=True)
    st = st.join(ns, on="country").with_columns([(pl.col(c).fill_null(0) / pl.col("ns") * 1000).alias(c) for c in ("s1f", "sin", "sout")]).drop("ns")
    st = st.with_columns(((pl.col("sin") >= IN_MIN) & (pl.col("sin") / (pl.col("sout") + 0.2) >= IO_MIN)).alias("var"))
    top = prep(sc.sort("p2", descending=True).unique("s23_id", keep="first"))
    top = top.join(st.select("country", pl.col("tok").alias("at"), "var", pl.col("s1f").alias("af")), on=["country", "at"], how="left").with_columns(pl.col("var").fill_null(False), pl.col("af").fill_null(0))
    top = top.with_columns(pl.struct("dt", "at").map_elements(lambda s: L.normalized_similarity(s["dt"] or "", s["at"] or ""), return_dtype=pl.Float64).alias("sim"),
                           (pl.col("p2") >= pl.col("country").replace_strict(T, default=0.75)).alias("acc"))
    one = pl.col("eq") & (pl.col("na") == 1) & (pl.col("nd") <= 1)
    vsel = one & pl.col("var") & (pl.col("p2") >= 0.01) & ~pl.col("acc")
    csel = (one & (pl.col("nd") == 1) & ~pl.col("var") & (pl.col("sim") < 0.5) & (pl.col("af") >= RW_MIN) & (pl.col("at").str.len_chars() >= 4)
            & ~pl.col("at").str.contains(r"\d") & ~pl.col("dt").str.starts_with(pl.col("at")) & pl.col("acc"))
    top = top.with_columns(vsel.alias("vsel"), csel.alias("csel"), ((pl.col("acc") | vsel) & ~csel).alias("acc2"))
    print(split, "vocab", st.filter("var").group_by("country").agg(pl.col("tok").sort()).sort("country").rows())
    return top
v = tops("train", pl.read_parquet(W + "valid_scores_stage2.parquet"))
gt = load_ground_truth(); s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
for c in ("acc", "acc2"):
    mm = macro_f05(s1v["s1_id"], v.filter(c).select("s1_id", "s23_id"), gt, by=s1v)
    print(f"val {c}: {mm['f05']:.5f} (US {mm['f05_US']:.5f} IN {mm['f05_India']:.5f})")
print("val vsel/csel", v.group_by("country").agg(pl.col("vsel").sum(), pl.col("label").filter("vsel").mean().alias("vprec"), pl.col("csel").sum(), pl.col("label").filter("csel").mean().alias("cprec")).rows())
t = tops("test", pl.read_parquet(W + "test_scores_stage2_v11.parquet"))
print("test vsel/csel", t.group_by("country").agg(pl.col("vsel").sum(), pl.col("csel").sum()).sort("country").rows())
def rd(f):
    m = pl.read_csv(f, separator="\t", schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
    return m.select(pl.col(m.columns[0]).alias("s1_id"), pl.col(m.columns[1]).str.split(",").alias("s23_id")).explode("s23_id").filter(pl.col("s23_id").str.len_chars() > 0)
vf = rd(W + "probe11_fr_vfix/matching_results.tsv")
mine = t.filter("acc2").select("s1_id", "s23_id", "country")
print("mine-not-vfix", mine.join(vf, on=["s1_id", "s23_id"], how="anti").group_by("country").len().sort("country").rows(),
      "vfix-not-mine", vf.join(mine.select("s1_id", "s23_id"), on=["s1_id", "s23_id"], how="anti").join(t.select("s1_id", "country").unique(), on="s1_id").group_by("country").len().rows())
for c in ("US", "India"):
    x = t.filter((pl.col("country") == c) & (pl.col("vsel") | pl.col("csel")))
    print(c, "vsel", x.filter("vsel").group_by("at").len().sort("len", descending=True).head(8).rows(), "csel", x.filter("csel").group_by("at", "dt").len().sort("len", descending=True).head(8).rows())
t.select("s1_id", "s23_id", "country", "p2", "acc", "vsel", "csel", "acc2").write_parquet(W + "vrule2_test.parquet")
