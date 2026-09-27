import cProfile, pstats, numpy as np, polars as pl
from src.features import load_side, pair_features
s1,q=load_side("train"); rng=np.random.default_rng(0); n=200_000
cand=pl.DataFrame({"q_row":rng.integers(0,len(q),n).astype(np.uint32),"s1_row":rng.integers(0,len(s1),n).astype(np.uint32)})
cProfile.run("pair_features(cand,s1,q)", "../work/prof.out")
pstats.Stats("../work/prof.out").sort_stats("tottime").print_stats(12)
