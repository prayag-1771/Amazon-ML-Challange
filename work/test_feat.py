import time, numpy as np, polars as pl
from src.features import load_side, pair_features
t=time.time(); s1,q=load_side("train"); print("load", f"{time.time()-t:.0f}s", s1.shape, q.shape, flush=True)
rng=np.random.default_rng(0); n=200_000
cand=pl.DataFrame({"q_row":rng.integers(0,len(q),n).astype(np.uint32),"s1_row":rng.integers(0,len(s1),n).astype(np.uint32)})
t=time.time(); f=pair_features(cand,s1,q); dt=time.time()-t
print(f"{n} pairs {dt:.1f}s -> {n/dt:,.0f}/s"); print(f.describe().transpose(include_header=True).head(40))
