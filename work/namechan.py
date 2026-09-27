"""Recall of a name-only char-ngram TF-IDF channel for empty-address queries (validation truth)."""
import sys, time; sys.path.insert(0, ".")
import numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from src.config import is_valid_expr
W = "../work/"
K = 20
s1 = pl.read_parquet(W+"norm_train_s1.parquet", columns=["entity_id","country","name_core"]).with_row_index("s1_row")
q = pl.concat([pl.read_parquet(W+f"norm_train_s{k}.parquet", columns=["entity_id","country","name_core","addr_empty"]) for k in (2,3)]).with_row_index("q_row")
gt = pl.read_parquet(W+"gt_pairs.parquet").filter(is_valid_expr("s1_id"))
qe = q.filter(pl.col("addr_empty")==1)
parts = []
for c in ("India","US"):
    a = s1.filter(pl.col("country")==c); b = qe.filter(pl.col("country")==c)
    t = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3), min_df=2, sublinear_tf=True, dtype=np.float32)
    X1 = vec.fit_transform(a["name_core"].to_list()).T.tocsr()
    Xq = vec.transform(b["name_core"].to_list())
    m = sp_matmul_topn(Xq, X1, top_n=K, threshold=0.1, sort=True, n_threads=14).tocoo()
    d = pl.DataFrame({"q_row": b["q_row"].to_numpy()[m.row], "s1_row": a["s1_row"].to_numpy()[m.col], "sim": m.data})
    d = d.with_columns(pl.col("sim").rank("ordinal", descending=True).over("q_row").alias("rk"))
    parts.append(d); print(c, len(b), X1.shape, f"{time.time()-t:.0f}s", flush=True)
nm = pl.concat(parts).with_columns(pl.col("q_row","s1_row").cast(pl.UInt32))
raw = pl.read_parquet(W+"cand_train.parquet", columns=["q_row","s1_row"]).with_columns(pl.lit(True).alias("raw"))
pr = pl.read_parquet(W+"pruned_train.parquet", columns=["q_row","s1_row"]).with_columns(pl.lit(True).alias("pr"))
ids1 = s1.select(pl.col("s1_row").cast(pl.UInt32), pl.col("entity_id").alias("s1_id"))
idq = qe.select(pl.col("q_row").cast(pl.UInt32), pl.col("entity_id").alias("s23_id"), "country")
g = gt.join(idq, on="s23_id").join(ids1, on="s1_id")
g = g.join(raw, on=["q_row","s1_row"], how="left").join(pr, on=["q_row","s1_row"], how="left").join(nm, on=["q_row","s1_row"], how="left")
g = g.with_columns(pl.col("raw","pr").fill_null(False))
for k in (5, 10, 20):
    print(k, g.group_by("country").agg(pl.len(), pl.col("raw").mean().alias("raw"), pl.col("pr").mean().alias("pruned"),
        (pl.col("rk")<=k).mean().alias("name_k"), ((pl.col("rk")<=k)|pl.col("pr")).mean().alias("pr_or_name"), ((pl.col("rk")<=k)|pl.col("raw")).mean().alias("raw_or_name")).sort("country"))
nm.write_parquet(W+"namechan_train_empty.parquet")
