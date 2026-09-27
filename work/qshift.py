import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
gt=load_ground_truth()
def stats(split,casc,scores,val):
    s1=pl.read_parquet(f'norm_{split}_s1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1_id'})
    if val: s1=s1.filter(is_valid_expr('s1_id'))
    c=pl.read_parquet(casc).join(s1,on='s1_id')
    sc=pl.read_parquet(scores).join(s1,on='s1_id')
    top=sc.sort('p2',descending=True).unique('s23_id',keep='first')
    n=s1.group_by('country').len('ns1')
    o=c.group_by('country').agg(pl.len().alias('pairs'),pl.col('s23_id').n_unique().alias('q')).join(n,on='country')
    o=o.join(top.group_by('country').agg((pl.col('p2')>=.75).sum().alias('acc'),(pl.col('p2').is_between(.1,.75,closed='left')).sum().alias('mid'),(pl.col('p2')<.1).sum().alias('low')),on='country')
    return o.with_columns([(pl.col(k)/pl.col('ns1')).round(3).alias(k+'/S1') for k in ('pairs','q','acc','mid','low')]).select('country','ns1','pairs/S1','q/S1','acc/S1','mid/S1','low/S1')
with pl.Config(tbl_cols=20):
    print('val',stats('train','cascade_valid_p001.parquet','valid_scores_stage2.parquet',True))
    print('test',stats('test','cascade_test_p001.parquet','test_scores_stage2_v15.parquet',False))
