import time, torch, polars as pl
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda"); m.half()
d = pl.read_parquet("../work/raw_train_s2.parquet").filter(pl.col("country")=="India").head(50000)
txt = ("query: " + d["business_name"] + " | " + d["business_address"]).to_list()
t=time.time(); e = m.encode(txt, batch_size=512, convert_to_tensor=True, normalize_embeddings=True); torch.cuda.synchronize()
print("50k in", f"{time.time()-t:.1f}s", e.shape, e.dtype)
