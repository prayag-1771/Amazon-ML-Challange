"""Simulate test's higher unmatched-query share on validation: replicate negative queries k times."""
import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
d = pl.read_parquet(W + "valid_scores_stage2.parquet")
neg = ~pl.col("s23_id").is_in(gt["s23_id"].implode())
nq = d.select("s23_id").unique()
print("val queries", len(nq), "negative share", d.filter(neg)["s23_id"].n_unique() / len(nq))
def sim(k, col, ts):
    parts = [d]
    dn = d.filter(neg)
    for r in range(1, k):
        parts.append(dn.with_columns((pl.col("s23_id") + f"_dup{r}").alias("s23_id")))
    dd = pl.concat(parts)
    top = dd.sort(col, descending=True).unique("s23_id", keep="first")
    share = dd.filter(neg | pl.col("s23_id").str.contains("_dup"))["s23_id"].n_unique() / dd["s23_id"].n_unique()
    out = []
    for t in ts:
        m = macro_f05(s1v["s1_id"], top.filter(pl.col(col) >= t).select("s1_id", "s23_id"), gt, by=s1v)
        out.append(f"t={t}: {m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f})")
    print(f"k={k} {col} negshare={share:.3f}\n  " + "\n  ".join(out), flush=True)
ts = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93]
for k in (1, 2, 3):
    sim(k, "p", ts); sim(k, "p2", ts)
