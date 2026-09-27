"""Per-segment threshold sweep for empty-address queries on top of pred15_v (stage-2 top-1)."""
import sys; sys.path.insert(0, "../business_entity_resolution")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
s1 = pl.read_parquet("norm_train_s1.parquet", columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt = load_ground_truth().join(s1.select("s1_id"), on="s1_id", how="semi")
q = pl.concat([pl.read_parquet(f"norm_train_s{k}.parquet", columns=["entity_id","addr_empty"]) for k in (2,3)]).rename({"entity_id":"s23_id"})
P = pl.read_parquet("pred15_v.parquet").select("s1_id","s23_id")
v = pl.read_parquet("valid_scores_stage2.parquet").join(s1, on="s1_id").join(q, on="s23_id")
top = v.sort("p2", descending=True).unique("s23_id", keep="first")
b0 = macro_f05(s1["s1_id"], P, gt, by=s1); print({k: round(x, 5) for k, x in b0.items() if k.startswith("f05")})
T = {"US": .75, "India": .8}
for e in (1, 0):
    for c in ("US", "India"):
        cur = top.filter((pl.col("addr_empty") == e) & (pl.col("country") == c))
        for t in ((.3, .4, .5, .6, .65, .7) if True else ()):
            if t >= T[c]: continue
            add = cur.filter((pl.col("p2") >= t) & (pl.col("p2") < T[c])).select("s1_id","s23_id").join(P, on="s23_id", how="anti")
            m = macro_f05(s1["s1_id"], pl.concat([P, add]), gt, by=s1)
            print("empty" if e else "addr", c, t, len(add), round(add.join(gt, on=["s1_id","s23_id"], how="semi").height / max(len(add), 1), 3), "dF_c %.5f" % (m["f05_"+c] - b0["f05_"+c]))
        for t in (.8, .85, .9):
            if t <= T[c]: continue
            rm = cur.filter((pl.col("p2") >= T[c]) & (pl.col("p2") < t)).select("s1_id","s23_id")
            m = macro_f05(s1["s1_id"], P.join(rm, on=["s1_id","s23_id"], how="anti"), gt, by=s1)
            print("empty" if e else "addr", c, "raise", t, len(rm), "dF_c %.5f" % (m["f05_"+c] - b0["f05_"+c]))
