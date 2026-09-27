import polars as pl
t=pl.read_parquet('frtok.parquet')
m=pl.read_csv('../output/matching_results.tsv',separator='\t').with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').drop_nulls().rename({'source1_entity_id':'s1_id','matched_entity_ids':'s23_id'}).with_columns(pl.lit(1).alias('acc'))
t=t.join(m,on=['s1_id','s23_id'],how='left').with_columns(pl.col('acc').fill_null(0))
x=t.filter((pl.col('n_out').list.len()==1)&(pl.col('n_in').list.len()==1)&(pl.col('a_out').list.len()==0)&(pl.col('a_in').list.len()==0))
x=x.with_columns(pl.col('n_out').list.first().alias('o'),pl.col('n_in').list.first().alias('i'))
with pl.Config(tbl_rows=40):
    print(x.group_by('bd').agg(pl.len(),pl.col('acc').mean()))
    print(x.filter(pl.col('bd')=='mid').group_by('i').agg(pl.len(),pl.col('acc').mean()).sort('len',descending=True).head(15))
    print(x.filter((pl.col('bd')=='mid')&(pl.col('i')=='fils')).select('p','p2','n1','nq','u1','uq','a1').head(12))
