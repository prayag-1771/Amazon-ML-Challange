import time, torch, numpy as np, polars as pl
from sentence_transformers import SentenceTransformer
from src.io_utils import load_ground_truth
gt = load_ground_truth()
m = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda"); m.half()
C = "India"
s1 = pl.read_parquet("../work/raw_train_s1.parquet").filter(pl.col("country")==C)
q = pl.concat([pl.read_parquet(f"../work/raw_train_s{i}.parquet") for i in (2,3)]).filter(pl.col("country")==C).sample(30000, seed=1)
g = gt.join(q.select(pl.col("entity_id").alias("s23_id")), on="s23_id", how="semi")
enc = lambda d: m.encode(("query: " + d["business_name"] + " | " + d["business_address"]).to_list(), batch_size=512, convert_to_tensor=True, normalize_embeddings=True)
t=time.time(); E1 = enc(s1); Eq = enc(q); print("enc", f"{time.time()-t:.0f}s", flush=True)
t=time.time(); idx=[]; 
for i in range(0, len(Eq), 4096):
    idx.append(torch.topk(Eq[i:i+4096] @ E1.T, 30, dim=1).indices.cpu())
idx = torch.cat(idx).numpy(); print("topk", f"{time.time()-t:.1f}s")
df = pl.DataFrame({"s23_id": np.repeat(q["entity_id"].to_numpy(), 30), "s1_id": s1["entity_id"].to_numpy()[idx.ravel()], "rk": np.tile(np.arange(1,31), len(q))})
hit = g.join(df, on=["s1_id","s23_id"], how="left")
print(C, "e5", " ".join(f"@{k}:{(hit['rk']<=k).sum()/len(g):.4f}" for k in (1,5,10,20,30)))
df.write_parquet("../work/bench_emb_india.parquet")
