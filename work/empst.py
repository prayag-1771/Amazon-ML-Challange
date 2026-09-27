import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.io_utils import load_ground_truth
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet') for k in (2,3)]).filter(pl.col('addr_empty')==1)
print(q.select('business_address','state','nums','addr_full','name_full').head(8))
print('state nonnull', q['state'].is_not_null().mean(), 'addr raw nonnull', q['business_address'].is_not_null().mean())
gt=load_ground_truth()
s1=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id','state']).rename({'entity_id':'s1_id','state':'st1'})
x=q.select(pl.col('entity_id').alias('s23_id'),'state').join(gt,on='s23_id').join(s1,on='s1_id')
print('state eq when both present', x.filter(pl.col('state').is_not_null()&pl.col('st1').is_not_null()).select((pl.col('state')==pl.col('st1')).mean()))
