"""Validate symmetry prior-shift correction: inflate up-shift negatives in validation (test-like prior), compare F0.5."""
import sys; sys.path.insert(0, ".")
import polars as pl
exec(open("../work/symcorr.py", encoding="utf-8").read().split("t, s1t = prep")[0].replace('print("val base"', 'print("val base0"').replace("for st in (0.5, 1.0):\n    for mn", "for st in ():\n    for mn"))
for M in (1, 5, 20, 50):
    neg = v.filter(pl.col("dg").is_not_null() & pl.col("up") & (pl.col("label") == 0))
    extra = pl.concat([neg.with_columns((pl.col("s23_id") + f"_r{k}").alias("s23_id")) for k in range(1, M)]) if M > 1 else neg.head(0)
    vs = pl.concat([v, extra])
    cs = prior(counts(vs), Rall)
    r = [f"M={M} base {f(vs, 'p2')}"]
    for st in (0.5, 1.0):
        r.append(f"st={st} {f(apply(vs, cs, st, 20), 'p3')}")
    print(" | ".join(r), flush=True)
