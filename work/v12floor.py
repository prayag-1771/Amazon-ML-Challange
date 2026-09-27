import sys; sys.path.insert(0, ".")
import polars as pl
import src.stage2 as S
W="../work/"
c=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country"]).rename({"entity_id":"s1_id"})
for tag in ("v11","v12"):
    d=pl.read_parquet(W+f"test_scores_stage2_{tag}.parquet").filter(pl.col("p")>=S.PCUT)
    top=d.sort("p2",descending=True).unique("s23_id",keep="first").join(c,on="s1_id")
    s1=pl.read_parquet(W+"norm_test_s1.parquet",columns=["entity_id","country","name_core","nums"]).rename({"entity_id":"s1_id","name_core":"n1","nums":"u1"})
    q=pl.concat([pl.read_parquet(W+f"norm_test_s{i}.parquet",columns=["entity_id","name_core","nums"]) for i in (2,3)]).rename({"entity_id":"s23_id","name_core":"nq","nums":"uq"})
    x=S._name_diff(top.filter(pl.col("country")=="France").select("s1_id","s23_id","p2"),s1.drop("country"),q)
    VFR=["fils","groupe","services","associes","developpement","france"]
    v=x.filter(pl.col("eq")&(pl.col("na")==1)&(pl.col("nd")<=1)&pl.col("at").is_in(VFR))
    print(tag,"vsel-shape top1",v.height,"p2 bins",v.group_by(pl.col("p2").cut([0.001,0.003,0.01,0.03,0.1,0.75])).len().sort("p2").rows())
