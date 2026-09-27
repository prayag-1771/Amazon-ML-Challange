import polars as pl, numpy as np, json
W="../work/"
c=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
t=pl.read_parquet(W+"test_feats.parquet").join(c,on="s1_id").join(pl.read_parquet(W+"test_scores_ens_v8.parquet"),on=["s1_id","s23_id"])
t=t.filter(pl.col("p")>0.99)
num=[k for k,v in t.schema.items() if v.is_numeric() and k not in("q_row","s1_row","p")]
rows=[]
for f in num:
    a=t.filter(pl.col("country")=="France")[f].to_numpy().astype(float); b=t.filter(pl.col("country")=="US")[f].to_numpy().astype(float); i=t.filter(pl.col("country")=="India")[f].to_numpy().astype(float)
    sd=np.nanstd(np.concatenate([b,i]))+1e-9
    rows.append((f,np.nanmean(a),np.nanmean(b),np.nanmean(i),min(abs(np.nanmean(a)-np.nanmean(b)),abs(np.nanmean(a)-np.nanmean(i)))/sd))
for r in sorted(rows,key=lambda r:-r[4])[:25]: print("%-22s FR %9.3f US %9.3f IN %9.3f  z %.2f"%r)
