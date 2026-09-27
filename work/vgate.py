import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
exec(open('frcell.py').read().split("rows=[]")[0])
s1,q=load('train'); s1=s1.filter(is_valid_expr('s1_id'))
v=prep(pl.read_parquet('valid_scores_stage2.parquet').sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id'))
v=v.filter((pl.col('na')==1)&(pl.col('nd')<=1)).with_columns(pl.col('A').list.first().alias('at'))
top=v.group_by('country','at').len().sort('len',descending=True).group_by('country').head(8)
print(top.sort('country','len').rows())
v=v.join(top.select('country','at',pl.lit(True).alias('var')),on=['country','at'],how='left').with_columns(pl.col('var').fill_null(False))
b=pl.col('p2').cut([.001,.01,.1,.75],labels=['<.001','.001','.01','.1','.75'])
print(v.filter('var').group_by('country',b.alias('b')).agg(pl.len(),pl.col('label').mean().round(3)).sort('country','b'))
