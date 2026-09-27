import sys; sys.path.insert(0, ".")
import polars as pl
import src.stage2 as s2
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W = "../work/"
s1v = pl.read_parquet(W + "norm_train_s1.parquet", columns=["entity_id", "country"]).filter(is_valid_expr("entity_id")).rename({"entity_id": "s1_id"})
gt = load_ground_truth()
v = pl.read_parquet(W + "valid_scores_stage2.parquet").sort("p2", descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id", "p2", "label").join(s1v, on="s1_id")
vt = s2.shift_cells(v, "train"); R, Rp = s2.shift_ratio(vt)
R.write_parquet(W + "shift_R.parquet"); Rp.write_parquet(W + "shift_Rp.parquet")
print(R.sort("country", "dg").rows(), Rp.sort("dg").rows())
c = pl.read_parquet(W + "norm_test_s1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
t = pl.read_parquet(W + "test_scores_stage2_v11.parquet").sort("p2", descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id", "p2").join(c, on="s1_id")
tt = s2.shift_cells(t, "test")
def exc(x):
    th = pl.col("country").replace_strict(s2.T, default=0.75)
    y = x.filter(pl.col("dg").is_not_null() & (pl.col("p2") >= th)).join(R, on=["country", "dg"], how="left").join(Rp.rename({"R": "Rp"}), on="dg", how="left").with_columns(pl.coalesce("R", "Rp").alias("R"))
    g = y.group_by("country", "dg").agg(pl.col("up").sum().alias("au"), (~pl.col("up")).sum().alias("ad"), pl.col("R").first())
    return g.group_by("country").agg((pl.col("au") - pl.col("R") * pl.col("ad")).sum().round(0)).sort("country").rows(), x.filter(pl.col("p2") >= th).group_by("country").len().sort("country").rows()
print("test no cap", exc(tt))
th = pl.col("country").replace_strict(s2.T, default=0.75)
for sl, z in ((1.0, 2.0), (1.2, 3.0), (1.5, 3.0)):
    vc = s2.shift_cap(vt, R, Rp, sl, z)
    mm = macro_f05(s1v["s1_id"], vc.filter(pl.col("p2") >= th).select("s1_id", "s23_id"), gt, by=s1v)
    print(f"slack={sl} z={z} val {mm['f05']:.5f} (US {mm['f05_US']:.5f} IN {mm['f05_India']:.5f})  test", exc(s2.shift_cap(tt, R, Rp, sl, z)), flush=True)
