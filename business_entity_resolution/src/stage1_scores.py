"""Stage-1 outputs that stage 2 and the add-on channels read (previously written by scripts under work/).

  python -m src.stage1_scores      after both first-stage models are trained and have predicted test
                                    (run_pipeline --model lgb_v5cf and --model lgb_v6k)

Writes to work/:
  valid_scores_{lgb_v5cf,lgb_v6k}.parquet   s1_id, s23_id, p, label  - first-stage scores of the validation pairs
  test_feats.parquet                          test pair features without the cross-encoder columns
  cascade_{valid,test}_p001.parquet           s1_id, s23_id with mean(v5cf, v6k) >= stage2.PCUT (the cascade set)
"""
import time

import polars as pl

from .config import WORK_DIR
from .model import load_model, predict

MODELS = ("lgb_v5cf", "lgb_v6k")
PCUT = 0.001  # = stage2.PCUT


def valid_scores():
    va = pl.read_parquet(WORK_DIR / "valid_feats.parquet")
    for name in MODELS:
        model, feats, _ = load_model(name)
        va.select("s1_id", "s23_id", "label").with_columns(pl.Series("p", predict(model, va, feats))).select(
            "s1_id", "s23_id", "p", "label").write_parquet(WORK_DIR / f"valid_scores_{name}.parquet")
        print(f"  valid_scores_{name}.parquet written", flush=True)


def test_feats():
    from .prune import prune
    from .run_pipeline import featurize  # imports torch (cross-encoder stage): only needed here
    path = WORK_DIR / "test_feats.parquet"
    if not path.exists():
        t = time.time()
        featurize(prune("test"), "test").write_parquet(path)
        print(f"  test_feats.parquet written {time.time() - t:.0f}s", flush=True)


def cascade_lists():
    for split in ("valid", "test"):
        a = pl.read_parquet(WORK_DIR / f"{split}_scores_{MODELS[0]}.parquet", columns=["s1_id", "s23_id", "p"]).rename({"p": "pa"})
        b = pl.read_parquet(WORK_DIR / f"{split}_scores_{MODELS[1]}.parquet", columns=["s1_id", "s23_id", "p"]).rename({"p": "pb"})
        d = a.join(b, on=["s1_id", "s23_id"]).filter((pl.col("pa") + pl.col("pb")) / 2 >= PCUT).select("s1_id", "s23_id")
        d.write_parquet(WORK_DIR / f"cascade_{split}_p001.parquet")
        print(f"  cascade_{split}_p001.parquet: {d.height:,} pairs", flush=True)


if __name__ == "__main__":
    valid_scores()
    test_feats()
    cascade_lists()
