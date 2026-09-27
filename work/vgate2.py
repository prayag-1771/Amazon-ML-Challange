import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(220)
exec(open('frcell.py').read().split("rows=[]")[0])
s1,q=load('train'); s1=s1.filter(is_valid_expr('s1_id'))
V={'US':['center','services','service','partners','lnc','trust','district','association'],'India':['center','services','service','partners','tek','lndia','5ervices','impeks']}
# all stage2 pairs (not only top-1) at same address with a single var swap
d=pl.read_parquet('valid_scores_stage2.parquet').join(s1,on='s1_id').join(q,on='s23_id')
d=prep(d).filter((pl.col('na')==1)&(pl.col('nd')==1)).with_columns(pl.col('A').list.first().alias('at'),pl.col('D').list.first().alias('dt'))
d=d.with_columns(pl.struct('country','at').map_elements(lambda r: r['at'] in V[r['country']],return_dtype=pl.Boolean).alias('var'))
f=s1.select('country',_tok('n1').alias('t')).explode('t').group_by('country','t').len('fr')
d=d.join(f.rename({'t':'dt'}),on=['country','dt'],how='left').with_columns(pl.col('fr').fill_null(0))
d=d.with_columns(pl.col('p2').rank('ordinal',descending=True).over('s23_id').alias('rk'))
print(d.group_by('country','var',pl.col('fr')>=100,'rk').agg(pl.len(),pl.col('label').mean().round(3),(pl.col('p2')>=.75).mean().round(3).alias('acc')).sort('country','var','fr','rk').filter(pl.col('rk')<=2))
print(d.filter(pl.col('var')&(pl.col('fr')>=100)&(pl.col('country')=='US')).sample(20,seed=1).select('nq','n1','label',pl.col('p2').round(3),'rk'))
