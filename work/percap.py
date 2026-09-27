import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.io_utils import load_ground_truth
gt=load_ground_truth()
s1=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1_id'})
def dist(pairs,s1,name):
    c=s1.join(pairs.group_by('s1_id').agg(pl.col('s23_id').str.starts_with('S2').sum().alias('a2'),pl.col('s23_id').str.starts_with('S3').sum().alias('a3')),on='s1_id',how='left').fill_null(0)
    c=c.with_columns(pl.max_horizontal('a2','a3').clip(0,7).alias('mx'))
    return c.group_by('country','mx').len().with_columns((pl.col('len')/pl.col('len').sum().over('country')*1000).round(2).alias(name)).drop('len')
T=dist(gt,s1,'truth')
st=pl.read_parquet('norm_test_s1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1_id'})
m=pl.read_csv('../output/matching_results.tsv',separator='\t',infer_schema_length=0).with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').drop_nulls().rename({'source1_entity_id':'s1_id','matched_entity_ids':'s23_id'})
A=dist(m,st,'test_acc')
with pl.Config(tbl_rows=40): print(A.join(T,on=['country','mx'],how='full',coalesce=True).sort('country','mx'))
