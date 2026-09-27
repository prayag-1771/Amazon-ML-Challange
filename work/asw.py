import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
tok=lambda c: pl.col(c).fill_null('').str.split(' ').list.eval(pl.element().filter(pl.element()!=''))
dig=lambda c: pl.col(c).str.replace_all(r'\D','')
def prep(split,sc,val):
    s1=pl.read_parquet(f'norm_{split}_s1.parquet',columns=['entity_id','country','name_core','addr_core']).rename({'entity_id':'s1_id','name_core':'n1','addr_core':'a1'})
    if val: s1=s1.filter(is_valid_expr('s1_id'))
    q=pl.concat([pl.read_parquet(f'norm_{split}_s{k}.parquet',columns=['entity_id','name_core','addr_core','addr_empty']) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','addr_core':'aq'})
    t=pl.read_parquet(sc).join(s1.select('s1_id'),on='s1_id').sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id').filter(pl.col('addr_empty')==0)
    t=t.with_columns(tok('n1').alias('tn1'),tok('nq').alias('tnq'),tok('a1').alias('ta1'),tok('aq').alias('taq'))
    t=t.filter((pl.col('tn1').list.sort()==pl.col('tnq').list.sort()))
    t=t.with_columns(pl.col('ta1').list.set_difference('taq').alias('o'),pl.col('taq').list.set_difference('ta1').alias('i'))
    t=t.filter((pl.col('o').list.len()==1)&(pl.col('i').list.len()==1)).with_columns(pl.col('o').list.first(),pl.col('i').list.first())
    t=t.with_columns((dig('o')==dig('i')).alias('samedig'),(dig('o')!='').alias('hasdig'))
    return t.with_columns(pl.col('p2').cut([.1,.3,.5,.75,.9],left_closed=True).alias('band'))
gt=load_ground_truth().with_columns(pl.lit(1).alias('lab'))
v=prep('train','valid_scores_stage2.parquet',True).join(gt,on=['s1_id','s23_id'],how='left').with_columns(pl.col('lab').fill_null(0))
t=prep('test','test_scores_stage2_v15.parquet',False)
m=pl.read_csv('../output/matching_results.tsv',separator='\t',infer_schema_length=0).with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').drop_nulls().rename({'source1_entity_id':'s1_id','matched_entity_ids':'s23_id'}).with_columns(pl.lit(1).alias('acc'))
t=t.join(m,on=['s1_id','s23_id'],how='left').with_columns(pl.col('acc').fill_null(0))
with pl.Config(tbl_rows=60,tbl_cols=10,fmt_str_lengths=30):
    print(v.group_by('country','samedig','hasdig','band').agg(pl.len(),pl.col('lab').mean().round(3)).sort('country','samedig','hasdig','band'))
    print(t.filter(pl.col('country')=='France').group_by('samedig','hasdig','band').agg(pl.len(),pl.col('acc').mean().round(3)).sort('samedig','hasdig','band'))
    print(t.filter((pl.col('country')=='France')&pl.col('samedig')&(pl.col('acc')==0)).group_by('o','i').len().sort('len',descending=True).head(25))
t.write_parquet('asw_t.parquet'); v.write_parquet('asw_v.parquet')
