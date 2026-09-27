"""Loss decomposition on a v15-like validation prediction (stage2 base + rescue + India addr channel), split by query addr_empty."""
import sys; sys.path.insert(0, "../business_entity_resolution")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
T = {"US": .75, "India": .8}
s1 = pl.read_parquet("norm_train_s1.parquet", columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt = load_ground_truth().join(s1.select("s1_id"), on="s1_id", how="semi")
v = pl.read_parquet("valid_scores_stage2.parquet").join(s1, on="s1_id")
top = v.sort("p2", descending=True).unique("s23_id", keep="first")
base = top.filter(pl.col("p2") >= pl.col("country").replace_strict(T)).select("s1_id","s23_id")
resc = pl.read_parquet("v10/rescue_valid.parquet").sort("pr", descending=True).unique("q_row", keep="first").filter(pl.col("vs1") & (pl.col("pr") >= .9)).join(v.select("s23_id").unique(), on="s23_id", how="anti")
ia = pl.read_parquet("v10/ac2_valid_mid.parquet").sort("pr", descending=True).unique("q_row", keep="first").filter(pl.col("vs1") & (pl.col("pr") >= .9)).join(v.select("s23_id").unique(), on="s23_id", how="anti")
pred = pl.concat([base, resc.select("s1_id","s23_id"), ia.select("s1_id","s23_id")]).unique()
pred.write_parquet("v10/val_pred15.parquet")
b0 = macro_f05(s1["s1_id"], pred, gt, by=s1); print("v15-like", {k: round(x, 5) for k, x in b0.items() if k.startswith("f05")})
q = pl.concat([pl.read_parquet(f"norm_train_s{k}.parquet", columns=["entity_id","addr_empty"]) for k in (2,3)]).rename({"entity_id":"s23_id"})
K = ["s1_id","s23_id"]
g = gt.join(s1, on="s1_id").join(q, on="s23_id")
g = g.with_columns(pl.struct(K).is_in(pred.select(pl.struct(K))["s1_id"] if False else pred.select(pl.struct(K)).to_series()).alias("hit"))
vq = v.select("s23_id").unique(); vp = v.select(K)
fn = g.filter(~pl.col("hit"))
fn = fn.with_columns(pl.col("s23_id").is_in(vq["s23_id"]).alias("qcand"), pl.struct(K).is_in(vp.select(pl.struct(K)).to_series()).alias("pcand"),
                     pl.col("s23_id").is_in(pred["s23_id"]).alias("qassigned"))
fn = fn.with_columns(pl.when(~pl.col("qcand")).then(pl.lit("noquerycand")).when(~pl.col("pcand")).then(pl.lit("notincand"))
                     .when(pl.col("qassigned")).then(pl.lit("wrongtop1")).otherwise(pl.lit("belowthr")).alias("cat"))
fp = pred.join(gt, on=K, how="anti").join(s1, on="s1_id").join(q, on="s23_id")
fp = fp.with_columns(pl.col("s23_id").is_in(gt["s23_id"]).alias("qtrue")).with_columns(pl.when(pl.col("qtrue")).then(pl.lit("FP_wrongS1")).otherwise(pl.lit("FP_singleton")).alias("cat"))
rows = []
for (c, e, cat), grp in fn.group_by("country","addr_empty","cat"):
    m = macro_f05(s1["s1_id"], pl.concat([pred, grp.select(K)]), gt, by=s1)
    rows.append((c, e, "FN_"+cat, len(grp), round(m["f05_"+c] - b0["f05_"+c], 5)))
for (c, e, cat), grp in fp.group_by("country","addr_empty","cat"):
    m = macro_f05(s1["s1_id"], pred.join(grp.select(K), on=K, how="anti"), gt, by=s1)
    rows.append((c, e, cat, len(grp), round(m["f05_"+c] - b0["f05_"+c], 5)))
for r in sorted(rows, key=lambda r: -r[4]): print(r)
fn.write_parquet("v10/val_fn15.parquet"); fp.write_parquet("v10/val_fp15.parquet")
