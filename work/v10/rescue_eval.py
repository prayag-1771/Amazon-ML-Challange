import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W = "../work/"
T = {"US": .75, "India": .8}
s1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt = load_ground_truth().join(s1.select("s1_id"), on="s1_id", how="semi")
v = pl.read_parquet(W+"valid_scores_stage2.parquet").join(s1, on="s1_id")
top = v.sort("p2", descending=True).unique("s23_id", keep="first")
base = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T)).select("s1_id","s23_id")
r = pl.read_parquet(W+"v10/rescue_valid.parquet").sort("pr", descending=True).unique("q_row", keep="first").filter(pl.col("vs1"))
r = r.join(base, on="s23_id", how="anti")   # only queries the pipeline leaves unassigned
b0 = macro_f05(s1["s1_id"], base, gt, by=s1); print("base", {k: round(x, 5) for k, x in b0.items() if k.startswith("f05")})
for t in (.5, .6, .7, .8, .9, .95):
    add = r.filter(pl.col("pr") >= t)
    m = macro_f05(s1["s1_id"], pl.concat([base, add.select("s1_id","s23_id")]), gt, by=s1)
    print(t, len(add), round(add["label"].mean(), 3), {k: round(x - b0[k], 5) for k, x in m.items() if k.startswith("f05")})
bc = base.group_by("s1_id").len("bcnt")
r = r.join(bc, on="s1_id", how="left").with_columns(pl.col("bcnt").fill_null(0))
qc = v.group_by("s23_id").agg(pl.col("p2").max().alias("qmax"))
r = r.join(qc, on="s23_id", how="left").with_columns(pl.col("qmax").fill_null(-1))
x = r.filter(pl.col("pr") >= .5)
print(x.group_by(pl.col("bcnt").clip(upper_bound=4), (pl.col("qmax") >= 0).alias("hascand"), (pl.col("pr") >= .9).alias("hi"), "is_in").agg(pl.len(), pl.col("label").mean().round(3)).sort("is_in","bcnt","hascand","hi"))
for name, f in [("bcnt>=1", pl.col("bcnt") >= 1), ("bcnt>=2", pl.col("bcnt") >= 2), ("nocand", pl.col("qmax") < 0), ("bcnt>=1&nocand", (pl.col("bcnt") >= 1) & (pl.col("qmax") < 0))]:
    for t in (.5, .7, .9):
        add = r.filter((pl.col("pr") >= t) & f)
        m = macro_f05(s1["s1_id"], pl.concat([base, add.select("s1_id","s23_id")]), gt, by=s1)
        print(name, t, len(add), round(add["label"].mean(), 3), {k: round(x - b0[k], 5) for k, x in m.items() if k.startswith("f05")})
