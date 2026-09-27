import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
gt=load_ground_truth().with_columns(pl.lit(1).alias('lab'))
def top(split,sc,val):
    s1=pl.read_parquet(f'norm_{split}_s1.parquet',columns=['entity_id','country','name_core','legal']).rename({'entity_id':'s1_id','name_core':'n1','legal':'l1'})
    if val: s1=s1.filter(is_valid_expr('s1_id'))
    q=pl.concat([pl.read_parquet(f'norm_{split}_s{k}.parquet',columns=['entity_id','addr_empty','name_core','legal']) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','legal':'lq'})
    t=pl.read_parquet(sc).join(s1.select('s1_id'),on='s1_id')
    t=t.with_columns(pl.len().over('s23_id').alias('nc'))
    t=t.sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id')
    # number of S1 in country with same name_core
    s1n=pl.read_parquet(f'norm_{split}_s1.parquet',columns=['country','name_core']).group_by('country','name_core').len('nsame')
    t=t.join(s1n,left_on=['country','n1'],right_on=['country','name_core'],how='left')
    return t.with_columns((pl.col('n1')==pl.col('nq')).alias('neq'))
v=top('train','valid_scores_stage2.parquet',True).join(gt,on=['s1_id','s23_id'],how='left').with_columns(pl.col('lab').fill_null(0))
t=top('test','test_scores_stage2_v15.parquet',False)
m=pl.read_csv('../output/matching_results.tsv',separator='\t').with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').drop_nulls().rename({'source1_entity_id':'s1_id','matched_entity_ids':'s23_id'}).with_columns(pl.lit(1).alias('acc'))
t=t.join(m,on=['s1_id','s23_id'],how='left').with_columns(pl.col('acc').fill_null(0))
f=lambda d:d.filter((pl.col('addr_empty')==1)&pl.col('neq')).with_columns(pl.col('p2').cut([.1,.3,.5,.75,.9],left_closed=True).alias('band'),(pl.col('nsame')==1).alias('uniq'))
with pl.Config(tbl_rows=60):
    print(f(v).group_by('country','uniq','band').agg(pl.len(),pl.col('lab').mean().round(3).alias('prec')).sort('country','uniq','band'))
    print(f(t).group_by('country','uniq','band').agg(pl.len(),pl.col('acc').mean().round(3).alias('accfrac')).sort('country','uniq','band'))
