import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
pl.Config.set_tbl_rows(60)
gt=load_ground_truth().with_columns(pl.lit(1).alias('lab'))
def top(split,sc,val):
    s1a=pl.read_parquet(f'norm_{split}_s1.parquet',columns=['entity_id','country','name_core','nums','legal','addr_core']).rename({'entity_id':'s1_id','name_core':'n1','nums':'u1','legal':'l1','addr_core':'a1'})
    s1a=s1a.join(s1a.group_by('country','a1').len('a1cnt'),on=['country','a1'])
    s1=s1a.filter(is_valid_expr('s1_id')) if val else s1a
    q=pl.concat([pl.read_parquet(f'norm_{split}_s{k}.parquet',columns=['entity_id','addr_empty','name_core','nums','legal','addr_core']) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','nums':'uq','legal':'lq','addr_core':'aq'})
    t=pl.read_parquet(sc).join(s1.select('s1_id'),on='s1_id')
    t=t.join(t.group_by('s23_id').agg(pl.col('p2').sort(descending=True).get(1,null_on_oob=True).alias('p2nd')),on='s23_id')
    t=t.sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id')
    a=pl.col('u1').str.split(' ').list.first(); b=pl.col('uq').str.split(' ').list.first()
    t=t.with_columns((pl.col('n1')==pl.col('nq')).alias('neq'),(a==b).fill_null(False).alias('numeq'),(pl.col('a1')==pl.col('aq')).alias('aeq'),(pl.col('l1').fill_null('')==pl.col('lq').fill_null('')).alias('leq'))
    return t
v=top('train','valid_scores_stage2.parquet',True).join(gt,on=['s1_id','s23_id'],how='left').with_columns(pl.col('lab').fill_null(0))
v.write_parquet('aeq_v.parquet')
t=top('test','test_scores_stage2_v15.parquet',False); t.write_parquet('aeq_t.parquet')
f=lambda d:d.filter(~pl.col('neq')&pl.col('numeq')&pl.col('aeq')&(pl.col('addr_empty')==0)&pl.col('p2').is_between(.05,.75)).with_columns(pl.col('a1cnt').clip(0,4),pl.col('p2').cut([.2,.4,.6]).alias('b'),pl.col('p2nd').fill_null(0).cut([.02,.1]).alias('b2'))
print(f(v).group_by('country','a1cnt','b').agg(pl.len(),pl.col('lab').mean().round(3)).sort('country','a1cnt','b'))
print(f(t).group_by('country','a1cnt','b').len().pivot(on='country',index=['a1cnt','b'],values='len').sort('a1cnt','b'))
