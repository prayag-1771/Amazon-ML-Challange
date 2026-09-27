"""Candidate generation: for every S2/S3 record (query) retrieve top-K S1 records.

Retrieval runs per country (matches never cross countries; country is only a partition key,
so unseen countries such as France work unchanged). Two channels, unioned:
  tf   word TF-IDF (rare tokens only, max_df 2%) on name_core + addr_core + state  (CPU)
  emb  multilingual-e5-small cosine on raw 'name | address'                       (GPU)
On a 30k-query India sample: emb@10 97.2%, tf@5 ~91%, union(10,5) 97.7% pair recall, ~13.5 cands/query.

Rows are referenced by position: s1_row into normalize_source(split, 1), q_row into
concat(normalize_source(split, 2), normalize_source(split, 3)).
"""
import sys
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .config import WORK_DIR
from .normalize import normalize_source

K_TF = 5
K_EMB = 10


def _frames(split: str):
    s1 = normalize_source(split, 1).select("entity_id", "country", "name_core", "addr_core", "state")
    q = pl.concat([normalize_source(split, s).select("entity_id", "country", "name_core", "addr_core", "state")
                   for s in (2, 3)])
    return s1.with_row_index("s1_row"), q.with_row_index("q_row")


def tf_text(d: pl.DataFrame) -> list:
    return (d["name_core"] + " " + d["addr_core"] + " " + d["state"]).to_list()


def _rank(q_idx: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """0-based rank of each (q, s1) pair within its query by descending sim."""
    order = np.lexsort((-sim, q_idx))
    qs = q_idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(qs)) + 1]
    pos = np.arange(len(qs)) - np.repeat(starts, np.diff(np.r_[starts, len(qs)]))
    rank = np.empty(len(qs), dtype=np.int16)
    rank[order] = pos
    return rank


def block_tfidf(split: str, k: int = K_TF, force: bool = False) -> pl.DataFrame:
    path = WORK_DIR / f"cand_tf_{split}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    s1, q = _frames(split)
    parts = []
    for country in sorted(s1["country"].unique().to_list()):
        a = s1.filter(pl.col("country") == country)
        b = q.filter(pl.col("country") == country)
        t = time.time()
        vec = TfidfVectorizer(analyzer="word", min_df=2, max_df=0.02, sublinear_tf=True, dtype=np.float32,
                              token_pattern=r"(?u)\b\w+\b")
        X1 = vec.fit_transform(tf_text(a)).T.tocsr()
        Xq = vec.transform(tf_text(b))
        for i in range(0, Xq.shape[0], 500_000):
            m = sp_matmul_topn(Xq[i:i + 500_000], X1, top_n=k, threshold=0.01, sort=True, n_threads=14).tocoo()
            parts.append(pl.DataFrame({
                "q_row": b["q_row"].to_numpy()[m.row + i], "s1_row": a["s1_row"].to_numpy()[m.col],
                "tf_sim": m.data.astype(np.float32), "tf_rank": _rank(m.row, m.data),
            }))
        print(f"  tf [{split}] {country}: S1={len(a):,} q={len(b):,} {time.time() - t:.0f}s", flush=True)
    out = pl.concat(parts)
    out.write_parquet(path)
    return out


def block_emb(split: str, k: int = K_EMB, force: bool = False, q_chunk: int = 2048) -> pl.DataFrame:
    import torch

    path = WORK_DIR / f"cand_emb_{split}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    s1, q = _frames(split)
    E1 = np.load(WORK_DIR / f"emb_{split}_s1.npy", mmap_mode="r")
    Eq = np.concatenate([np.load(WORK_DIR / f"emb_{split}_s{s}.npy") for s in (2, 3)])
    parts = []
    for country in sorted(s1["country"].unique().to_list()):
        a = s1.filter(pl.col("country") == country)["s1_row"].to_numpy()
        b = q.filter(pl.col("country") == country)["q_row"].to_numpy()
        t = time.time()
        idx = torch.from_numpy(np.ascontiguousarray(E1[a])).cuda()
        sims, cols = [], []
        with torch.no_grad():
            for i in range(0, len(b), q_chunk):
                qe = torch.from_numpy(Eq[b[i:i + q_chunk]]).cuda()
                v, c = torch.topk(qe @ idx.T, k, dim=1)
                sims.append(v.float().cpu())
                cols.append(c.int().cpu())
        del idx
        torch.cuda.empty_cache()
        sims = torch.cat(sims).numpy()
        cols = torch.cat(cols).numpy()
        parts.append(pl.DataFrame({
            "q_row": np.repeat(b, k).astype(np.uint32), "s1_row": a[cols.ravel()].astype(np.uint32),
            "emb_sim": sims.ravel(), "emb_rank": np.tile(np.arange(k, dtype=np.int16), len(b)),
        }))
        print(f"  emb [{split}] {country}: S1={len(a):,} q={len(b):,} {time.time() - t:.0f}s", flush=True)
    out = pl.concat(parts)
    out.write_parquet(path)
    return out


def build_candidates(split: str, force: bool = False) -> pl.DataFrame:
    """Union of channels: q_row, s1_row, tf_sim/rank, emb_sim/rank (null when not retrieved)."""
    path = WORK_DIR / f"cand_{split}.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    tf = block_tfidf(split).with_columns(pl.col("q_row", "s1_row").cast(pl.UInt32))
    emb = block_emb(split)
    out = emb.join(tf, on=["q_row", "s1_row"], how="full", coalesce=True).sort("q_row")
    out.write_parquet(path)
    return out


def with_ids(cand: pl.DataFrame, split: str) -> pl.DataFrame:
    s1, q = _frames(split)
    return (
        cand.join(s1.select(pl.col("s1_row").cast(pl.UInt32), pl.col("entity_id").alias("s1_id")), on="s1_row")
        .join(q.select(pl.col("q_row").cast(pl.UInt32), pl.col("entity_id").alias("s23_id")), on="q_row")
    )


if __name__ == "__main__":
    stage = sys.argv[1]
    for split in sys.argv[2:]:
        {"tf": block_tfidf, "emb": block_emb, "union": build_candidates}[stage](split, force=True)
