import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
s1=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id','country']).filter(is_valid_expr('entity_id')).rename({'entity_id':'s1_id'})
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','addr_empty']) for k in (2,3)]).rename({'entity_id':'s23_id'})
gt=load_ground_truth().join(s1,on='s1_id').join(q,on='s23_id')
fn=pl.read_parquet('fn15.parquet')
t=gt.group_by('country','addr_empty').len().join(fn.group_by('country','addr_empty').len(),on=['country','addr_empty'],suffix='_fn').with_columns((1-pl.col('len_fn')/pl.col('len')).alias('recall'))
print(t.sort('country','addr_empty'))
# test: all S1 present, so fraction of queries matched is comparable to train truth fraction
