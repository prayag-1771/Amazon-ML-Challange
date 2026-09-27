import sys
sys.path.insert(0, ".")
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(220)
W = "../work/"
def nums(split, s):
    return pl.read_parquet(W+f"norm_{split}_s{s}.parquet", columns=["entity_id","nums","country"] if s==1 else ["entity_id","nums"])
def rel(a, b):
    if not a or not b: return "none"
    A = [x.lstrip("0") or "0" for x in a.split()]; B = [x.lstrip("0") or "0" for x in b.split()]
    if A[0] == B[0]: return "first_eq"
    if set(A) & set(B): return "shared"
    x, y = A[0], B[0]
    if x in y or y in x: return "substr"
    if len(x) == len(y):
        d = sum(c1 != c2 for c1, c2 in zip(x, y))
        if d == 1: return "sub1"
        if sorted(x) == sorted(y): return "transp"
    if abs(len(x)-len(y)) == 1:
        s, l = (x, y) if len(x) < len(y) else (y, x)
        if any(l[:i]+l[i+1:] == s for i in range(len(l))): return "del1"
    return "diff"
for split in ("val", "test"):
    if split == "val":
        d = pl.read_parquet(W+"valid_scores_lgb_v6k.parquet"); tr = "train"
    else:
        d = pl.read_parquet(W+"test_scores_lgb_v6k.parquet").with_columns(pl.lit(None, pl.Int8).alias("label")); tr = "test"
    top = d.sort("p", descending=True).unique("s23_id", keep="first").filter(pl.col("p") > 0.3)
    s1 = nums(tr, 1).rename({"entity_id":"s1_id","nums":"n1"})
    q = pl.concat([nums(tr, 2), nums(tr, 3)]).rename({"entity_id":"s23_id","nums":"nq"})
    n1 = (pl.read_parquet(W+"valid_feats.parquet", columns=["s1_id"]).unique().join(s1, on="s1_id") if split=="val" else s1).group_by("country").len("n1")
    top = top.join(s1, on="s1_id").join(q, on="s23_id")
    top = top.with_columns(pl.struct("n1","nq").map_elements(lambda r: rel(r["n1"], r["nq"]), return_dtype=pl.Utf8).alias("rel"),
                           pl.col("p").cut([0.7,0.95,0.999]).alias("b"))
    g = top.group_by("country","rel","b").agg(pl.len().alias("n"), pl.col("label").mean().alias("y")).join(n1, on="country").with_columns((pl.col("n")/pl.col("n1")*1000).round(2).alias("per1k"))
    print(split); print(g.filter(~pl.col("rel").is_in(["first_eq","none"])).sort("country","rel","b").select("country","rel","b","per1k","y","n"))
    top.write_parquet(W+f"numrel_{split}.parquet")
