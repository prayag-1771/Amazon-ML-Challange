import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
gt=load_ground_truth().with_columns(pl.lit(1).alias('lab'))
def top(split,sc,val):
    s1=pl.read_parquet(f'norm_{split}_s1.parquet',columns=['entity_id','country','name_core','nums','legal','addr_core']).rename({'entity_id':'s1_id','name_core':'n1','nums':'u1','legal':'l1','addr_core':'a1'})
    if val: s1=s1.filter(is_valid_expr('s1_id'))
    q=pl.concat([pl.read_parquet(f'norm_{split}_s{k}.parquet',columns=['entity_id','addr_empty','name_core','nums','legal','addr_core']) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','nums':'uq','legal':'lq','addr_core':'aq'})
    t=pl.read_parquet(sc).join(s1.select('s1_id'),on='s1_id')
    t=t.sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id')
    a=pl.col('u1').str.split(' ').list.first(); b=pl.col('uq').str.split(' ').list.first()
    t=t.with_columns((pl.col('n1')==pl.col('nq')).alias('neq'),(a==b).fill_null(False).alias('numeq'),(pl.col('a1')==pl.col('aq')).alias('aeq'),(pl.col('l1').fill_null('')==pl.col('lq').fill_null('')).alias('leq'))
    return t, s1.group_by('country').len()
v,nv=top('train','valid_scores_stage2.parquet',True)
v=v.join(gt,on=['s1_id','s23_id'],how='left').with_columns(pl.col('lab').fill_null(0))
t,nt=top('test','test_scores_stage2_v15.parquet',False)
bins=[.75,.9,.95,.98,.99]
K=['country','addr_empty','band','neq','numeq','aeq','leq']
f=lambda d:d.filter(pl.col('p2')>=.75).with_columns(pl.col('p2').cut(bins,left_closed=True).alias('band'))
A=f(v).group_by(K).agg(pl.len().alias('nv'),pl.col('lab').sum().alias('posv')).join(nv,on='country').with_columns((pl.col('nv')/pl.col('len')*1e4).alias('cv'),(pl.col('posv')/pl.col('len')*1e4).alias('pv')).drop('len')
B=f(t).group_by(K).agg(pl.len().alias('nt')).join(nt,on='country').with_columns((pl.col('nt')/pl.col('len')*1e4).alias('ct')).drop('len')
X=B.join(A,on=K,how='left').with_columns((pl.col('posv')/pl.col('nv')).round(3).alias('precV'),(pl.col('ct')/pl.col('cv')).round(2).alias('ratio'),(pl.col('ct')-pl.col('cv')).round(1).alias('excess_per1e4'))
X=X.filter(pl.col('country')!='France').sort('excess_per1e4',descending=True)
with pl.Config(tbl_rows=40,tbl_cols=20): print(X.select(K+['nt','nv','ratio','precV','excess_per1e4']).head(30))
