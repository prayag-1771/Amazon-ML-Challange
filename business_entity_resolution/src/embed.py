"""Multilingual sentence embeddings (intfloat/multilingual-e5-small, MIT) of 'name | address'.

Embeddings are L2-normalised fp16, one .npy per split/source, row-aligned with load_source().
Raw (not normalised) text is embedded so the model still sees native scripts.
"""
import sys
import time

import numpy as np
import polars as pl

from .config import WORK_DIR
from .io_utils import load_source

MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK = 500_000


def record_text(df: pl.DataFrame) -> list:
    return ("query: " + df["business_name"] + " | " + df["business_address"]).to_list()


def load_model(device: str = "cuda"):
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(MODEL_NAME, device=device)
    m.half()
    return m


def embed_source(split: str, source: int, model=None, force: bool = False) -> np.ndarray:
    path = WORK_DIR / f"emb_{split}_s{source}.npy"
    if path.exists() and not force:
        return np.load(path, mmap_mode="r")
    model = model or load_model()
    txt = record_text(load_source(split, source))
    out = np.empty((len(txt), model.get_sentence_embedding_dimension()), dtype=np.float16)
    t = time.time()
    for i in range(0, len(txt), CHUNK):
        e = model.encode(txt[i:i + CHUNK], batch_size=512, normalize_embeddings=True, convert_to_numpy=True)
        out[i:i + CHUNK] = e.astype(np.float16)
        print(f"  {split} s{source}: {min(i + CHUNK, len(txt)):,}/{len(txt):,}  {time.time() - t:.0f}s", flush=True)
    np.save(path, out)
    return out


if __name__ == "__main__":
    model = load_model()
    for split in sys.argv[1:] or ["train", "test"]:
        for src in (1, 2, 3):
            embed_source(split, src, model)
