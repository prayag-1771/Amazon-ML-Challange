import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src import stage2 as S
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
s1=pl.read_parquet('norm_test_s1.parquet',columns=['entity_id','country','name_core','nums']).rename({'entity_id':'s1_id','name_core':'n1','nums':'u1'})
q=pl.concat([pl.read_parquet(f'norm_test_s{i}.parquet',columns=['entity_id','name_core','nums']) for i in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','nums':'uq'})
d=pl.read_parquet('test_scores_stage2_v15.parquet')
t0=S._name_diff(d.sort('p',descending=True).unique('s23_id',keep='first').select('s1_id','s23_id'),s1,q)
sw=t0.filter((pl.col('na')==1)&(pl.col('nd')==1)&pl.col('eq'))
st=s1.select('country',S._tok('n1').alias('tok')).explode('tok').drop_nulls().group_by('country','tok').len('s1f')
for z in (sw.group_by('country',pl.col('at').alias('tok')).len('sin'),sw.group_by('country',pl.col('dt').alias('tok')).len('sout')):
    st=st.join(z,on=['country','tok'],how='full',coalesce=True)
st=st.join(s1.group_by('country').len('ns'),on='country').with_columns([(pl.col(c).fill_null(0)/pl.col('ns')*1000).round(2).alias(c) for c in ('s1f','sin','sout')])
st=st.with_columns((pl.col('sin')/(pl.col('sout')+.2)).round(1).alias('io'))
for c in ('France','US'):
    print(c); print(st.filter(pl.col('country')==c).sort('sin',descending=True).head(25).select('tok','s1f','sin','sout','io'))
