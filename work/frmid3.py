import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
def load(split):
    s1=pl.read_parquet(f'norm_{split}_s1.parquet',columns=['entity_id','country','name_core','addr_empty']).rename({'entity_id':'s1_id','name_core':'n1','addr_empty':'e1'})
    q=pl.concat([pl.read_parquet(f'norm_{split}_s{k}.parquet',columns=['entity_id','name_core','addr_empty']) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','addr_empty':'eq_'})
    nc=s1.group_by('country','n1').len('ncnt')
    return s1.join(nc,on=['country','n1']),q
# test
s1,q=load('test')
d=pl.read_parquet('test_scores_stage2_v15.parquet')
top=d.sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id')
acc=pl.read_parquet('../work/sub15_indiaaddr/matching_results.tsv' if False else 'pred15_t.parquet') if False else None
import csv
rows=[]
with open('sub15_indiaaddr/matching_results.tsv') as f:
    next(f)
    for l in f:
        a,b=l.rstrip('\n').split('\t')
        if b: rows+= [(a,x) for x in b.split(',')]
M=pl.DataFrame(rows,schema=['s1_id','s23_id'],orient='row')
top=top.join(M.with_columns(pl.lit(1).alias('acc')),on=['s1_id','s23_id'],how='left').with_columns(pl.col('acc').fill_null(0))
top=top.with_columns((pl.col('n1')==pl.col('nq')).alias('neq'))
g=top.filter(pl.col('p2').is_between(.1,.75)).group_by('country','eq_','neq',pl.col('ncnt')==1).agg(pl.len(),pl.col('acc').mean().round(3),pl.col('p2').median().round(2)).sort('country','eq_','neq','ncnt')
print('TEST');print(g)
# valid
s1,q=load('train'); s1=s1.filter(is_valid_expr('s1_id'))
gt=load_ground_truth()
v=pl.read_parquet('valid_scores_stage2.parquet')
vt=v.sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id').with_columns((pl.col('n1')==pl.col('nq')).alias('neq'))
vt=vt.join(pl.read_parquet('pred15_v.parquet').select('s1_id','s23_id',pl.lit(1).alias('acc')),on=['s1_id','s23_id'],how='left').with_columns(pl.col('acc').fill_null(0))
g=vt.filter(pl.col('p2').is_between(.1,.75)).group_by('country','eq_','neq',pl.col('ncnt')==1).agg(pl.len(),pl.col('label').mean().round(3),pl.col('acc').mean().round(3),pl.col('p2').median().round(2)).sort('country','eq_','neq','ncnt')
print('VALID');print(g)
