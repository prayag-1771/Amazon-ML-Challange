import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
bands=[0,.01,.05,.1,.2,.3,.4,.5,.6,.75]
def band(c): return pl.col(c).cut(bands[1:],labels=[str(b) for b in bands]).alias('band')
# validation
s1v=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id','country']).filter(is_valid_expr('entity_id')).rename({'entity_id':'s1_id'})
P=pl.read_parquet('pred15_v.parquet')
cv=P.group_by('s1_id').len('nacc')
v=pl.read_parquet('valid_scores_stage2.parquet').sort('p2',descending=True).unique('s23_id',keep='first').join(s1v,on='s1_id')
v=v.join(P.select('s23_id',pl.lit(1).alias('acc')),on='s23_id',how='left').filter(pl.col('acc').is_null())
v=v.join(cv,on='s1_id',how='left').with_columns(pl.col('nacc').fill_null(0))
print('VALID global', s1v.join(cv,on='s1_id',how='left').group_by('country').agg(pl.col('nacc').fill_null(0).mean().round(3)).rows())
print(v.with_columns(band('p2')).group_by('country','band','label').agg(pl.len(),pl.col('nacc').mean().round(3)).sort('country','band','label').pivot(on='label',index=['country','band'],values=['len','nacc']))
# test
rows=[]
with open('sub15_indiaaddr/matching_results.tsv') as f:
    next(f)
    for l in f:
        a,b=l.rstrip('\n').split('\t')
        if b: rows+=[(a,x) for x in b.split(',')]
M=pl.DataFrame(rows,schema=['s1_id','s23_id'],orient='row')
s1=pl.read_parquet('norm_test_s1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1_id'})
ct=M.group_by('s1_id').len('nacc')
print('TEST global', s1.join(ct,on='s1_id',how='left').group_by('country').agg(pl.col('nacc').fill_null(0).mean().round(3)).rows())
t=pl.read_parquet('test_scores_stage2_v15.parquet').sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id')
t=t.join(M.select('s23_id',pl.lit(1).alias('acc')),on='s23_id',how='left').filter(pl.col('acc').is_null()).join(ct,on='s1_id',how='left').with_columns(pl.col('nacc').fill_null(0))
print(t.with_columns(band('p2')).group_by('country','band').agg(pl.len(),pl.col('nacc').mean().round(3)).sort('country','band').pivot(on='country',index='band',values=['len','nacc']))
