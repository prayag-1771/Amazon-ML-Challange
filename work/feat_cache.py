"""Cache test pair features (without the cross-encoder columns) for fast re-prediction and France experiments."""
import sys
import time

sys.path.insert(0, ".")
from src.config import WORK_DIR
from src.prune import prune
from src.run_pipeline import featurize

t = time.time()
df = featurize(prune("test"), "test")
df.write_parquet(WORK_DIR / "test_feats.parquet")
print(f"test feats {df.shape} {time.time() - t:.0f}s")
