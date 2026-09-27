import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.metric import macro_f05
from src.io_utils import load_ground_truth
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
d = pl.read_parquet(W + "valid_scores_stage2.parquet")
top = d.sort("p2", descending=True).unique("s23_id", keep="first").join(s1v, on="s1_id")
ing = top["s23_id"].is_in(gt["s23_id"].implode())
top = top.with_columns(ing.alias("has_gt"))
print("queries/S1", top.group_by("country").agg(pl.len(), pl.col("has_gt").mean()).join(s1v.group_by("country").len("n"), on="country").with_columns((pl.col("len")/pl.col("n")).alias("q_per_s1")))
un = top.filter(~pl.col("has_gt"))
print("unmatched top p2 dist", un.select([(pl.col("p2") >= t).mean().alias(str(t)) for t in (0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)]))
for w in (1, 3, 5):
    extra = pl.concat([un.with_columns((pl.col("s23_id") + f"_r{k}").alias("s23_id")) for k in range(1, w)]) if w > 1 else un.head(0)
    tt = pl.concat([top, extra])
    r = []
    for t in (0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        m = macro_f05(s1v["s1_id"], tt.filter(pl.col("p2") >= t).select("s1_id", "s23_id"), gt, by=s1v)
        r.append(f"t={t} {m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f})")
    print(f"w={w}", " | ".join(r), flush=True)
