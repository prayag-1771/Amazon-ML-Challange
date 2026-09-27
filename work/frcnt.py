import polars as pl
t=pl.read_parquet('frcell_t.parquet')
rows=[]
with open('sub15_indiaaddr/matching_results.tsv') as f:
    next(f)
    for l in f:
        a,b=l.rstrip('\n').split('\t'); rows.append((a,len(b.split(',')) if b else 0))
C=pl.DataFrame(rows,schema=['s1_id','nacc'],orient='row')
s1=pl.read_parquet('norm_test_s1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1_id'})
C=C.join(s1,on='s1_id')
print(C.group_by('country').agg(pl.col('nacc').mean().round(3),(pl.col('nacc')==0).mean().round(4).alias('z')).sort('country'))
print(C.group_by('country','nacc').len().pivot(on='country',index='nacc',values='len').sort('nacc'))
f=t.filter((pl.col('country')=='France')&(pl.col('na')==1)&(pl.col('nd')==1))
for acc in (0,1):
    x=f.filter(pl.col('acc')==acc).select('s1_id').unique().join(C,on='s1_id')
    print('acc',acc,x.height,x['nacc'].mean(),x['nacc'].value_counts().sort('nacc').rows())
