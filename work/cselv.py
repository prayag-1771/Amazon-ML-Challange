import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src import stage2 as S
s1v=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1_id'})
d=pl.read_parquet('valid_scores_stage2.parquet')
top=d.sort('p2',descending=True).unique('s23_id',keep='first').join(s1v,on='s1_id')
out=S.variant_rule(top.select('s1_id','s23_id','p2','country'),d,'train',['US','India'])
x=top.join(out.select('s23_id',pl.col('p2').alias('p2r')),on='s23_id')
th=pl.col('country').replace_strict(S.T,default=.75)
rej=x.filter((pl.col('p2')>=th)&(pl.col('p2r')<th)); acc=x.filter((pl.col('p2')<th)&(pl.col('p2r')>=th))
print('rejected',rej.group_by('country').agg(pl.len(),pl.col('label').mean().round(3)).rows())
print('accepted',acc.group_by('country').agg(pl.len(),pl.col('label').mean().round(3)).rows())
