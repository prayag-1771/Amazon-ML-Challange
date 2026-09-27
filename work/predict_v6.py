"""Predict test with a saved model from cached test features (work/test_feats.parquet) plus cross-encoder
columns, then write outputs exactly like run_pipeline.stage_predict."""
import sys
import time

sys.path.insert(0, ".")
import polars as pl

import src.run_pipeline as rp

rp.MODEL_NAME = sys.argv[1]
rp.CROSS_FIT = True
t = time.time()
src_feats = rp.WORK_DIR / "test_feats.parquet"
orig_prune, orig_feat = rp.prune, rp.featurize
rp.prune = lambda split: pl.read_parquet(src_feats) if split == "test" else orig_prune(split)
rp.featurize = lambda cand, split, **kw: cand if split == "test" else orig_feat(cand, split, **kw)
rp.stage_predict()
print(f"done {time.time() - t:.0f}s")
