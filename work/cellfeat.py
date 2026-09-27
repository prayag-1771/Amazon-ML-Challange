import sys; sys.path.insert(0, "."); sys.path.insert(0, "../work")
import polars as pl
import src.stage2 as s2
from symfix import prep
from src.config import is_valid_expr
cols = ["p", "pa", "pb", "ce_logit", "nc_tset", "ac_tset", "q_addr_empty", "s1_name_cnt", "q_n", "s1_n", "s1_hi_same", "s1_hi_other", "alt_hi_tot", "sib_num", "sib_name", "sib_add_max", "add_n", "drop_n"]
v, _ = prep("valid_scores_stage2.parquet", "train", is_valid_expr("s1_id"))
t, _ = prep("test_scores_stage2_v10.parquet", "test")
cell = (pl.col("dg") == "d3") & (pl.col("leg") == "add") & pl.col("nsame") & pl.col("up")
dv = s2.sib_feats(s2.struct_feats(s2.feats(s2.base("valid")), "train"), "train")
vk = v.filter(cell & (pl.col("country") == "US")).select("s1_id", "s23_id", "label", "p2").join(dv.drop("label"), on=["s1_id", "s23_id"])
dt = s2.sib_feats(s2.struct_feats(s2.feats(s2.base("test")), "test"), "test")
tk = t.filter(cell & (pl.col("country") == "US")).select("s1_id", "s23_id", "p2").join(dt, on=["s1_id", "s23_id"])
tk.write_parquet("../work/cell_us_d3add_test.parquet"); vk.write_parquet("../work/cell_us_d3add_val.parquet")
rows = []
for c in cols + ["p2"]:
    rows.append((c, round(vk.filter(pl.col("label") == 0)[c].mean(), 3), round(vk.filter(pl.col("label") == 1)[c].mean(), 3), round(tk[c].mean(), 3), round(tk.filter(pl.col("p2") >= .75)[c].mean(), 3)))
print("feature  val_neg  val_pos  test_all  test_acc")
for r in rows: print(r)
