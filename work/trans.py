import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.io_utils import load_ground_truth
pl.Config.set_tbl_rows(40)
pred=pl.read_parquet("pred15_v.parquet")
gt=load_ground_truth()
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','country','name_core','addr_core','addr_empty']).with_columns(pl.lit(k).alias("src")) for k in (2,3)]).rename({'entity_id':'s23_id'})
# restrict to queries whose true S1 is valid or singleton: use queries not matched to a non-valid S1 in gt
vs1=set(pred["s1_id"].unique().to_list())
nonval=gt.filter(~pl.col("s1_id").is_in(list(set(gt["s1_id"].unique().to_list())-vs1)) )
qv=q.join(gt.filter(~pl.col("s1_id").is_in(pred["s1_id"].unique().implode())).select("s23_id"),on="s23_id",how="anti")  # drop queries of train-only S1s
un=qv.join(pred.select("s23_id"),on="s23_id",how="anti")
acc=pred.join(q,on="s23_id")
for key in (["country","name_core"],["country","name_core","addr_core"]):
    fam=acc.group_by(key).agg(pl.col("s1_id").unique().alias("s1s"))
    fam=fam.filter(pl.col("s1s").list.len()==1).with_columns(pl.col("s1s").list.first().alias("s1_id")).drop("s1s")
    c=un.join(fam,on=key).join(gt.with_columns(pl.lit(1).alias("y")),on=["s1_id","s23_id"],how="left").fill_null(0)
    print(key, c.group_by("country","addr_empty").agg(pl.len(),pl.col("y").mean().round(3)))
