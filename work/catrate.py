import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
tok=lambda c: pl.col(c).fill_null('').str.split(' ').list.eval(pl.element().filter(pl.element()!='')).list.unique()
def feats(pairs,split):
    s1=pl.read_parquet(f'norm_{split}_s1.parquet',columns=['entity_id','country','name_core','nums','legal']).rename({'entity_id':'s1_id','name_core':'n1','nums':'u1','legal':'l1'})
    q=pl.concat([pl.read_parquet(f'norm_{split}_s{k}.parquet',columns=['entity_id','addr_empty','name_core','nums','legal']) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','nums':'uq','legal':'lq'})
    d=pairs.join(s1,on='s1_id').join(q,on='s23_id')
    a=pl.col('u1').str.split(' ').list.first().cast(pl.Int64,strict=False); b=pl.col('uq').str.split(' ').list.first().cast(pl.Int64,strict=False)
    d=d.with_columns((b-a).alias('sh'),tok('n1').alias('t1'),tok('nq').alias('tq'))
    d=d.with_columns(pl.col('tq').list.set_difference('t1').list.len().clip(0,2).alias('na'),pl.col('t1').list.set_difference('tq').list.len().clip(0,2).alias('nd'))
    d=d.with_columns(pl.when(pl.col('addr_empty')==1).then(pl.lit('E')).when(pl.col('sh').is_null()).then(pl.lit('nonum')).when(pl.col('sh')==0).then(pl.lit('0')).when(pl.col('sh').abs()<=2).then(pl.lit('±12')).otherwise(pl.lit('far')).alias('num'),
                     (pl.col('l1').fill_null('')==pl.col('lq').fill_null('')).alias('leq'))
    return d.select('country','num','na','nd','leq')
gt=load_ground_truth()
s1v=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id','country']).filter(is_valid_expr('entity_id')).rename({'entity_id':'s1_id'})
tv=feats(gt.join(s1v.select('s1_id'),on='s1_id'),'train')
m=pl.read_csv('../output/matching_results.tsv',separator='\t',infer_schema_length=0).with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').drop_nulls().rename({'source1_entity_id':'s1_id','matched_entity_ids':'s23_id'})
ta=feats(m,'test')
s1t=pl.read_parquet('norm_test_s1.parquet',columns=['entity_id','country']).group_by('country').len('ns')
K=['num','na','nd','leq']
T=tv.group_by(['country']+K).len('n').join(s1v.group_by('country').len('ns'),on='country').with_columns((pl.col('n')/pl.col('ns')*1000).alias('r')).pivot(on='country',index=K,values='r')
A=ta.group_by(['country']+K).len('n').join(s1t,on='country').with_columns((pl.col('n')/pl.col('ns')*1000).alias('r')).pivot(on='country',index=K,values='r')
X=T.rename({'US':'US_true','India':'IN_true'}).join(A.rename({'US':'US_acc','India':'IN_acc','France':'FR_acc'}),on=K,how='full',coalesce=True).fill_null(0)
X=X.with_columns(((pl.col('US_true')+pl.col('IN_true'))/2).alias('avg_true')).with_columns((pl.col('FR_acc')-pl.col('avg_true')).alias('FR_minus'),(pl.col('US_acc')-pl.col('US_true')).alias('US_minus'),(pl.col('IN_acc')-pl.col('IN_true')).alias('IN_minus'))
with pl.Config(tbl_rows=100,tbl_cols=20,float_precision=1):
    print(X.sort(pl.col('FR_minus').abs(),descending=True).head(40))
    print(X.select(pl.col(['US_true','IN_true','US_acc','IN_acc','FR_acc']).sum()))
