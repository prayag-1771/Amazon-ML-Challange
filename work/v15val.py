import sys; sys.path.insert(0, ".")
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
W="../work/"; T={"US":.75,"India":.8}
s1=pl.read_parquet(W+"norm_train_s1.parquet",columns=["entity_id","country"]).filter(is_valid_expr("entity_id")).rename({"entity_id":"s1_id"})
gt=load_ground_truth().join(s1.select("s1_id"),on="s1_id",how="semi")
v=pl.read_parquet(W+"valid_scores_stage2.parquet").join(s1,on="s1_id")
top=v.sort("p2",descending=True).unique("s23_id",keep="first")
base=top.filter(pl.col("p2")>=pl.col("country").replace_strict(T)).select("s1_id","s23_id")
r=pl.read_parquet(W+"v10/rescue_valid.parquet").filter(pl.col("vs1")).sort("pr",descending=True).unique("s23_id",keep="first").filter(pl.col("pr")>=.9).join(v.select("s23_id").unique(),on="s23_id",how="anti")
b14=pl.concat([base,r.select("s1_id","s23_id")])
casc=pl.read_parquet(W+"cascade_valid_p001.parquet").select("s23_id").unique()
a=pl.read_parquet(W+"v10/ac2_valid_mid.parquet").sort("pr",descending=True).unique("q_row",keep="first").filter(pl.col("vs1")).join(casc,on="s23_id",how="anti").join(b14,on="s23_id",how="anti")
m0=macro_f05(s1["s1_id"],b14,gt,by=s1); print("v14",{k:round(x,5) for k,x in m0.items()})
for t in (.9,):
    add=a.filter(pl.col("pr")>=t); m=macro_f05(s1["s1_id"],pl.concat([b14,add.select("s1_id","s23_id")]),gt,by=s1)
    print(t,len(add),round(add["label"].mean(),3),{k:round(x,5) for k,x in m.items()})
