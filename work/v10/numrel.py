"""Number-noise pattern of first house numbers for name-equal/swap1 top-1 pairs with differing numbers.
US/India val split by label; France test split by accepted. Also predicted-empty S1 rate per country."""
import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=220)
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
W = "../work/"
exec(open("../work/v10/frcat.py").read().split("def load")[0].split("W = \"../work/\"")[1])
from rapidfuzz.distance import Levenshtein as L
def rel(a, b):
    if a is None or b is None: return "none"
    if len(a) == len(b) and sorted(a) == sorted(b): return "transp"
    d = L.distance(a, b)
    if d == 1: return "ed1_" + ("sub" if len(a) == len(b) else "indel")
    try:
        x = int(b) - int(a)
    except: return "nonint"
    if 0 < abs(x) <= 25: return "shift25"
    if d == 2: return "ed2"
    return "far"
def load(split):
    s1 = pl.read_parquet(W + f"norm_{split}_s1.parquet", columns=["entity_id", "country", "name_core", "nums"]).rename({"entity_id": "s1_id", "name_core": "n1", "nums": "u1"})
    q = pl.concat([pl.read_parquet(W + f"norm_{split}_s{i}.parquet", columns=["entity_id", "name_core", "nums"]) for i in (2, 3)]).rename({"entity_id": "s23_id", "name_core": "nq", "nums": "uq"})
    return s1, q
def go(top):
    x = cats(top).filter(pl.col("nm").is_in(["name=", "swap1"]) & (pl.col("num") == "num!="))
    return x.with_columns(pl.struct("a1", "aq").map_elements(lambda s: rel(s["a1"], s["aq"]), return_dtype=pl.String).alias("nr"))
T = 0.75
s1, q = load("test")
d = pl.read_parquet(W + "test_scores_stage2_v9.parquet")
top = d.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
acc = top.filter(pl.col("p2") >= T).group_by("s1_id").len()
print("TEST predicted-empty S1 rate", s1.join(acc, on="s1_id", how="left").group_by("country").agg(pl.col("len").is_null().mean()))
nfr = s1.filter(pl.col("country") == "France").height
fr = go(top.filter(pl.col("country") == "France"))
print("FRANCE", fr.group_by("nm", "nr").agg((pl.len() / nfr).round(4).alias("q"), ((pl.col("p2") >= T).sum() / nfr).round(4).alias("acc")).sort("nm", "q", descending=True))
s1, q = load("train")
v = pl.read_parquet(W + "valid_scores_stage2.parquet")
vt = v.sort("p2", descending=True).unique("s23_id", keep="first").join(s1, on="s1_id").join(q, on="s23_id")
s1v = s1.filter(is_valid_expr("s1_id"))
acc = vt.filter(pl.col("p2") >= T).group_by("s1_id").len()
print("VAL predicted-empty S1 rate", s1v.join(acc, on="s1_id", how="left").group_by("country").agg(pl.col("len").is_null().mean()))
gt = load_ground_truth()
print("VAL true singleton rate", s1v.join(gt.select("s1_id").unique().with_columns(pl.lit(1).alias("h")), on="s1_id", how="left").group_by("country").agg(pl.col("h").is_null().mean()))
for c in ("US", "India"):
    n = s1v.filter(pl.col("country") == c).height
    x = go(vt.filter(pl.col("country") == c))
    print(c, x.group_by("nm", "nr").agg((pl.len() / n).round(4).alias("q"), (pl.col("label").sum() / n).round(4).alias("true"), ((pl.col("p2") >= T).sum() / n).round(4).alias("acc")).sort("nm", "q", descending=True))
