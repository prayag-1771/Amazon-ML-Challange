import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
T={'US':.75,'India':.8}
s1=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id','country','nums']).filter(is_valid_expr('entity_id')).rename({'entity_id':'s1_id','nums':'u1'})
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','nums']) for k in (2,3)]).rename({'entity_id':'s23_id','nums':'uq'})
gt=load_ground_truth().with_columns(pl.lit(1).alias('label'))
v=pl.read_parquet('valid_scores_stage2.parquet').join(s1,on='s1_id')
top=v.sort('p2',descending=True).unique('s23_id',keep='first').filter(pl.col('p2')>=pl.col('country').replace_strict(T)).join(q,on='s23_id').join(gt,on=['s1_id','s23_id'],how='left').with_columns(pl.col('label').fill_null(0))
a=pl.col('u1').str.split(' ').list.first().cast(pl.Int64,strict=False); b=pl.col('uq').str.split(' ').list.first().cast(pl.Int64,strict=False)
x=top.with_columns((b-a).alias('d')).filter(pl.col('d').abs().is_between(1,12))
with pl.Config(tbl_rows=40): print(x.group_by('country','d').agg(pl.len(),pl.col('label').mean().round(3)).sort('country','d').filter(pl.col('country')=='India'))
g=gt.join(s1,on='s1_id').join(q,on='s23_id').with_columns((b-a).alias('d')).filter(pl.col('d').abs().is_between(1,3))
print('TRUTH India/US by d', g.group_by('country','d').len().sort('country','d'))
