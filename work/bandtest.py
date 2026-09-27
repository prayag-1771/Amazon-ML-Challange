import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
s1v=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id','country']).filter(is_valid_expr('entity_id')).rename({'entity_id':'s1_id'})
qv=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=['entity_id','addr_empty']) for k in (2,3)]).rename({'entity_id':'s23_id'})
gt=load_ground_truth().with_columns(pl.lit(1).alias('lab'))
v=pl.read_parquet('valid_scores_stage2.parquet').join(s1v,on='s1_id')
vt=v.sort('p2',descending=True).unique('s23_id',keep='first').join(qv,on='s23_id').join(gt,on=['s1_id','s23_id'],how='left').with_columns(pl.col('lab').fill_null(0))
s1t=pl.read_parquet('norm_test_s1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1_id'})
tt=pl.read_parquet('top15_test.parquet')
bins=[.5,.6,.7,.75,.8,.85,.9,.95,.98,.99,1.01]
def band(df):
    return df.with_columns(pl.col('p2').cut(bins[:-1],left_closed=True).alias('band'))
nv=s1v.group_by('country').len(); nt=s1t.group_by('country').len()
a=band(vt).filter(pl.col('p2')>=.5).group_by('country','addr_empty','band').agg(pl.len().alias('nv'),pl.col('lab').sum().alias('posv')).join(nv,on='country').with_columns((pl.col('nv')/pl.col('len')).alias('cv'),(pl.col('posv')/pl.col('len')).alias('pv')).drop('len')
b=band(tt).filter(pl.col('p2')>=.5).group_by('country','addr_empty','band').agg(pl.len().alias('nt')).join(nt,on='country').with_columns((pl.col('nt')/pl.col('len')).alias('ct')).drop('len')
x=b.join(a,on=['country','addr_empty','band'],how='left').with_columns((pl.col('posv')/pl.col('nv')).round(3).alias('precV'),(pl.col('pv')/pl.col('ct')).round(3).alias('precT_est'),(pl.col('ct')/pl.col('cv')).round(2).alias('ratio')).sort('country','addr_empty','band')
with pl.Config(tbl_rows=80): print(x.select('country','addr_empty','band','nt','nv','ratio','precV','precT_est'))
