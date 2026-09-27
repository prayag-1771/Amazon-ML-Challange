"""Stage-2 CV experiments: + raw pair features, + capacity. Single seed for speed."""
import sys; sys.path.insert(0, ".")
import numpy as np, polars as pl, lightgbm as lgb
import src.stage2 as S
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W = "../work/"
EX = ['emb_sim','tf_sim','emb_rank','tf_rank','nc_ratio','nc_jw','nc_tsort','nc_partial','nf_ratio','alt_ratio','nc_eq','legal_eq',
      'ac_ratio','ac_partial','ac_tsort','state_eq','city_jacc','addr_tok_jacc','num_jacc','num_first_eq','num_min_rel','q_nums_n',
      's1_nums_n','s1_name_state_cnt','q_name_in_s1','xq_rmin','xq_relmin','x1_rmin','x1_relmin','xq_fmax','x1_fmax','nc_nospace','nc_len1','nc_len2']
s1v = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt = load_ground_truth()
d = S.ce_hard_feats(S.sib_feats(S.struct_feats(S.feats(S.base("valid")), "train"), "train"), "valid")
d = d.join(pl.read_parquet(W+"valid_feats.parquet", columns=["s1_id","s23_id"]+EX), on=["s1_id","s23_id"], how="left")
qf = d.filter(pl.col("q_rank")==1).select("s23_id", (pl.col("s1_id").hash(3) % 5).alias("fold"))
d = d.join(qf, on="s23_id").join(s1v, on="s1_id")
y = d["label"].to_numpy(); fold = d["fold"].to_numpy()
th = pl.col("country").replace_strict(S.T, default=0.75)
def cv(F, P, R, name):
    X = d.select(F).to_numpy().astype(np.float32); oof = np.zeros(len(d), np.float32)
    for k in range(5):
        tr, te = fold != k, fold == k
        oof[te] = lgb.train(P, lgb.Dataset(X[tr], y[tr]), R).predict(X[te], num_threads=14)
    x = d.select("s1_id","s23_id","country").with_columns(pl.Series("p2", oof))
    top = x.sort("p2", descending=True).unique("s23_id", keep="first")
    res = []
    for dt in (-0.1, -0.05, 0, 0.05):
        m = macro_f05(s1v["s1_id"], top.filter(pl.col("p2") >= th + dt).select("s1_id","s23_id"), gt, by=s1v)
        res.append(f"dt{dt:+.2f} {m['f05']:.5f} (US {m['f05_US']:.5f} IN {m['f05_India']:.5f})")
    print(name, " | ".join(res), flush=True)
    return oof
P = S.P; R = S.R
cv(S.F, P, R, "base      ")
cv(S.F + EX, P, R, "+raw      ")
o = cv(S.F + EX, {**P, "num_leaves": 127, "min_data_in_leaf": 100, "learning_rate": 0.03}, 1200, "+raw big  ")
d.select("s1_id","s23_id","label").with_columns(pl.Series("p2", o)).write_parquet(W+"s2exp_oof_big.parquet")
