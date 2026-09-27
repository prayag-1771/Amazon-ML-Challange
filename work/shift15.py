import polars as pl
def load(p,c):
    return pl.read_csv(p,separator='\t').with_columns(pl.col(c).str.split(',')).explode(c).drop_nulls().rename({c:'s23_id','source1_entity_id':'s1_id'})
m=load('sub15_indiaaddr/matching_results.tsv','matched_entity_ids')
s1=pl.read_parquet('norm_test_s1.parquet',columns=['entity_id','country','nums','name_core']).rename({'entity_id':'s1_id','nums':'u1','name_core':'n1'})
q=pl.concat([pl.read_parquet(f'norm_test_s{k}.parquet',columns=['entity_id','nums','name_core']) for k in (2,3)]).rename({'entity_id':'s23_id','nums':'uq','name_core':'nq'})
x=m.join(s1,on='s1_id').join(q,on='s23_id')
a=pl.col('u1').str.split(' ').list.first().cast(pl.Int64,strict=False); b=pl.col('uq').str.split(' ').list.first().cast(pl.Int64,strict=False)
x=x.with_columns((b-a).alias('d'),(pl.col('n1')==pl.col('nq')).alias('ns'))
t=x.filter(pl.col('d').abs().is_between(1,25)).group_by('country','d').len()
up=t.filter(pl.col('d')>0); dn=t.filter(pl.col('d')<0).with_columns(-pl.col('d'))
r=up.join(dn,on=['country','d'],suffix='_dn').with_columns(pl.col('len','len_dn').cast(pl.Int64)).with_columns((pl.col('len')-pl.col('len_dn')).alias('excess')).sort('country','d')
with pl.Config(tbl_rows=100): print(r.pivot(on='country',index='d',values='excess')); print(r.filter(pl.col('d')<=12).pivot(on='country',index='d',values=['len','len_dn']))
print(r.filter(pl.col('d')>2).group_by('country').agg(pl.col('excess').sum()))
