import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
exec(open('frname.py').read().split("s1,q=load('test')")[0])
def sh(t):
    f=lambda c: pl.col(c).fill_null("").str.split(" ").list.first().cast(pl.Int64,strict=False)
    return t.filter(~pl.col('n1eq')).with_columns((f('uq')-f('u1')).alias('dlt')).filter(pl.col('dlt').is_not_null())
s1,q=load('test')
t=sh(prep(pl.read_parquet('test_scores_stage2_v15.parquet').sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id'))).filter(pl.col('uniq'))
for c in ('France','US','India'):
    x=t.filter(pl.col('country')==c)
    g=x.group_by('dlt').agg(pl.len(),(pl.col('p2')>=.75).mean().round(2).alias('hi')).sort('len',descending=True).head(16)
    print(c,x.height,g.rows())
s1,q=load('train'); s1=s1.filter(is_valid_expr('s1_id'))
v=sh(prep(pl.read_parquet('valid_scores_stage2.parquet').sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id'))).filter(pl.col('uniq'))
for c in ('US','India'):
    x=v.filter(pl.col('country')==c)
    g=x.group_by('dlt').agg(pl.len(),pl.col('label').mean().round(2).alias('lab'),(pl.col('p2')>=.75).mean().round(2).alias('hi')).sort('len',descending=True).head(16)
    print('V',c,x.height,g.rows())
