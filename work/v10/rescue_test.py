"""Apply the empty-address name rescue to test (US/India only): char-3gram top-10 -> rescue.lgb -> top-1 with pr>=.9."""
import sys, time; sys.path.insert(0, ".")
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from rapidfuzz.distance import JaroWinkler
W = "../work/"
s1 = pl.read_parquet(W+"norm_test_s1.parquet", columns=["entity_id","country","name_core","legal","addr_empty"]).with_row_index("s1_row").with_columns(pl.col("s1_row").cast(pl.UInt32))
q = pl.concat([pl.read_parquet(W+f"norm_test_s{k}.parquet", columns=["entity_id","country","name_core","legal","addr_empty"]) for k in (2,3)]).with_row_index("q_row").with_columns(pl.col("q_row").cast(pl.UInt32))
qe = q.filter(pl.col("addr_empty")==1)
parts = []
for c in ("India","US"):
    a = s1.filter(pl.col("country")==c); b = qe.filter(pl.col("country")==c); t = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3), min_df=2, sublinear_tf=True, dtype=np.float32)
    X1 = vec.fit_transform(a["name_core"].to_list()).T.tocsr(); Xq = vec.transform(b["name_core"].to_list())
    m = sp_matmul_topn(Xq, X1, top_n=10, threshold=0.1, sort=True, n_threads=14).tocoo()
    d = pl.DataFrame({"q_row": b["q_row"].to_numpy()[m.row], "s1_row": a["s1_row"].to_numpy()[m.col], "sim": m.data})
    parts.append(d.with_columns(pl.col("sim").rank("ordinal", descending=True).over("q_row").alias("rk"))); print(c, len(b), f"{time.time()-t:.0f}s", flush=True)
nm = pl.concat(parts).with_columns(pl.col("q_row","s1_row").cast(pl.UInt32))
s1 = s1.join(s1.group_by("country","name_core").len("s1_ncnt"), on=["country","name_core"])
d = nm.join(s1.select("s1_row", pl.col("entity_id").alias("s1_id"), "country", pl.col("name_core").alias("n1"), pl.col("legal").alias("l1"), "s1_ncnt", pl.col("addr_empty").alias("s1_empty")), on="s1_row")
d = d.join(q.select("q_row", pl.col("entity_id").alias("s23_id"), pl.col("name_core").alias("nq"), pl.col("legal").alias("lq")), on="q_row")
d = d.join(d.group_by("country","nq").agg(pl.col("q_row").n_unique().alias("q_ncnt")), on=["country","nq"])
d = d.with_columns(
    (pl.col("sim") - pl.col("sim").max().over("q_row")).alias("gap_best"),
    (pl.col("sim") - pl.col("sim").top_k(2).min().over("q_row")).alias("gap_2nd"),
    (pl.col("sim") >= pl.col("sim").max().over("q_row") - 0.02).sum().over("q_row").alias("n_tie"),
    pl.len().over("q_row").alias("n_c"), (pl.col("n1") == pl.col("nq")).cast(pl.Int8).alias("eq"),
    pl.when((pl.col("l1") == "") | (pl.col("lq") == "")).then(-1).otherwise((pl.col("l1") == pl.col("lq")).cast(pl.Int8)).alias("leq"),
    (pl.col("country") == "India").cast(pl.Int8).alias("is_in"))
n1, n2 = d["n1"].to_list(), d["nq"].to_list()
d = d.with_columns(pl.Series("r", cpdist(n1, n2, scorer=fuzz.ratio, workers=-1, dtype=np.float32)),
                   pl.Series("ts", cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                   pl.Series("jw", cpdist(n1, n2, scorer=JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)),
                   pl.col("n1").str.len_chars().alias("len1"), pl.col("nq").str.len_chars().alias("lenq"))
d = d.with_columns((pl.col("r") - pl.col("r").max().over("q_row")).alias("r_gap"),
                   (pl.col("r") >= pl.col("r").max().over("q_row") - 3).sum().over("q_row").alias("r_ntie"), pl.len().over("s1_row").alias("s1_nq"))
F = ["sim","rk","gap_best","gap_2nd","n_tie","n_c","eq","leq","is_in","s1_ncnt","q_ncnt","s1_empty","r","ts","jw","len1","lenq","r_gap","r_ntie","s1_nq"]
d = d.with_columns(pl.Series("pr", lgb.Booster(model_file=W+"v10/rescue.lgb").predict(d.select(F).to_numpy().astype(np.float32))))
top = d.sort("pr", descending=True).unique("q_row", keep="first").filter(pl.col("pr") >= .9)
# only queries with no candidate at all in the v12 cascade set
cas = pl.read_parquet(W+"cascade_test_p001.parquet", columns=["s23_id"]).unique()
top = top.join(cas, on="s23_id", how="anti")
top.select("s1_id","s23_id","country","pr","n1","nq").write_parquet(W+"v10/rescue_test_accept.parquet")
print(top.group_by("country").agg(pl.len(), pl.col("pr").mean()))
