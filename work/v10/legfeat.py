import sys; sys.path.insert(0, ".")
import polars as pl, lightgbm as lgb, numpy as np
from src import stage2 as S
W = "../work/"
t = pl.read_parquet(W + "v10/fr_legsib.parquet").filter(pl.col("st") & (pl.col("leg") == "add")).select("s1_id", "s23_id", "acc", "sg")
d = S.ce_hard_feats(S.sib_feats(S.struct_feats(S.feats(S.base("test")), "test"), "test"), "test")
x = d.join(t, on=["s1_id", "s23_id"])
m = lgb.Booster(model_file=W + "stage2.lgb")
X = x.select(S.F).to_numpy()
x = x.with_columns(pl.Series("p2", m.predict(X, num_threads=14)))
contrib = m.predict(X, pred_contrib=True, num_threads=14)[:, :-1]
rej = ~x["acc"].to_numpy()
mc = contrib[rej].mean(0) - contrib[~rej].mean(0)
print("mean SHAP diff rejected-accepted:")
for f, v in sorted(zip(S.F, mc), key=lambda z: z[1])[:12]: print(f"  {f:14s} {v:+.3f}")
print(x.group_by("acc").agg([pl.col(f).mean().round(3) for f in S.F]).transpose(include_header=True))
x.write_parquet(W + "v10/legfeat.parquet")
