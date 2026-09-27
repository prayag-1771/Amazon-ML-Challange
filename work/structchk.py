import sys; sys.path.insert(0, "."); sys.path.insert(0, "../work")
import polars as pl, lightgbm as lgb
import src.stage2 as s2
from symfix import prep, K, T
F9 = s2.F[:24]; FS = s2.F[:27]
dt = s2.struct_feats(s2.feats(s2.base("test")), "test")
m9 = lgb.Booster(model_file="../work/stage2_v9.lgb"); ms = lgb.Booster(model_file="../work/stage2_v10struct.lgb")
print(m9.num_feature(), ms.num_feature())
dt = dt.with_columns(pl.Series("p9", m9.predict(dt.select(F9).to_numpy(), num_threads=14)), pl.Series("ps", ms.predict(dt.select(FS).to_numpy(), num_threads=14)))
dt.select("s1_id", "s23_id", "p", "p9", "ps").write_parquet("../work/test_scores_stage2_struct.parquet")
t, _ = prep("test_scores_stage2_v10.parquet", "test")
t = t.select("s23_id", "country", "dg", "leg", "nsame", "up")
th = pl.col("country").replace_strict(T)
for c in ("p9", "ps"):
    top = dt.sort(c, descending=True).unique("s23_id", keep="first").select("s1_id", "s23_id", c).join(t, on="s23_id")
    x = top.filter(pl.col("dg").is_not_null() & pl.col("up"))
    print(c, "up-shift acc", x.group_by("country").agg((pl.col(c) >= th).sum()).sort("country").rows(), "total", top.group_by("country").agg((pl.col(c) >= th).sum()).sort("country").rows())
    print(x.group_by(K).agg((pl.col(c) >= th).sum().alias("a")).sort("a", descending=True).head(8).rows())
