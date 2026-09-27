import polars as pl
t=pl.read_parquet('frtok.parquet')
z=t.filter((pl.col('n_out').list.len()==0)&(pl.col('n_in').list.len()==0)&(pl.col('a_out').list.len()==0)&(pl.col('a_in').list.len()==0))
s1=pl.read_parquet('norm_test_s1.parquet',columns=['entity_id','country','name_core','addr_core']).filter(pl.col('country')=='France')
tw=s1.group_by('name_core','addr_core').len('ntw')
z=z.join(tw,left_on=['n1','a1'],right_on=['name_core','addr_core'],how='left')
tc=pl.read_parquet('test_scores_stage2_v15.parquet').group_by('s23_id').len('nc')
z=z.join(tc,on='s23_id')
with pl.Config(tbl_rows=30,fmt_str_lengths=40,tbl_cols=12):
    print(z.group_by('bd').agg(pl.len(),pl.col('ntw').mean(),(pl.col('ntw')>1).mean().alias('twin'),pl.col('nc').mean(),pl.col('leq').mean(),pl.col('numeq').mean(),pl.col('addr_empty').mean()))
    print(z.filter(pl.col('bd')=='mid').select('p','p2','n1','l1','lq','u1','uq','a1','aq','ntw','nc').sample(15,seed=3))
m=pl.read_csv('../output/matching_results.tsv',separator='\t',infer_schema_length=0).with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').drop_nulls().rename({'source1_entity_id':'s1_id','matched_entity_ids':'s23_id'}).with_columns(pl.lit(1).alias('acc'))
z=z.join(m,on=['s1_id','s23_id'],how='left').with_columns(pl.col('acc').fill_null(0))
with pl.Config(tbl_rows=40):
    print(z.filter(pl.col('bd')!='hi').group_by('bd','acc','leq').agg(pl.len(),pl.col('p').mean(),(pl.col('p')>=.9).mean().alias('p09')).sort('bd','acc','leq'))
    print(z.filter((pl.col('acc')==0)).group_by('l1','lq').len().sort('len',descending=True).head(12))
    print(z.filter((pl.col('acc')==0)&pl.col('leq')).select('p','p2','n1','l1','lq','a1','aq').head(10))
