import polars as pl
t=pl.read_parquet('top15_test.parquet')
f=t.filter(pl.col('country')=='France')
f=f.with_columns(pl.col('p2').cut([.1,.3,.5,.75,.9],left_closed=True).alias('band'))
with pl.Config(tbl_rows=80,tbl_cols=12):
    print(f.group_by('addr_empty','band').len().sort('addr_empty','band'))
    print(f.filter(pl.col('p2')<.75).group_by('addr_empty','neq','numeq','aeq','leq').len().sort('len',descending=True).head(20))
# queries count France
q=pl.concat([pl.read_parquet(f'norm_test_s{k}.parquet',columns=['entity_id','country','addr_empty']) for k in (2,3)]).filter(pl.col('country')=='France')
acc=pl.read_csv('../output/matching_results.tsv',separator='\t') if False else None
print('France queries',len(q),'with cand',f.height,'addr_empty frac',q['addr_empty'].mean())
s1=pl.read_parquet('norm_test_s1.parquet',columns=['country']).group_by('country').len(); print(s1)
