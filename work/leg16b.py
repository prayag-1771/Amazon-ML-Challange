import polars as pl
x=pl.read_parquet('leg16_t.parquet')
s1=pl.read_parquet('norm_test_s1.parquet',columns=['entity_id','country','name_core','legal','addr_core','business_name']).filter(pl.col('country')=='US')
# other S1s with same name_core + addr_core
sib=x.select('s1_id','s23_id','n1','a1','lq').join(s1.rename({'entity_id':'sid'}),left_on=['n1','a1'],right_on=['name_core','addr_core'])
print('pairs',len(x),'sib rows',len(sib), 'queries with >1 S1 sharing name+addr', sib.group_by('s23_id').len().filter(pl.col('len')>1).height)
print('queries where some sibling S1 has legal == lq', sib.filter(pl.col('legal').fill_null('')==pl.col('lq').fill_null('')).select('s23_id').n_unique())
t=pl.read_parquet('test_scores_stage2_v15.parquet')
cc=t.join(x.select('s23_id'),on='s23_id').group_by('s23_id').agg(pl.len().alias('nc'),pl.col('p2').sort(descending=True).get(1,null_on_oob=True).alias('p2nd'))
print(cc.describe())
q=pl.concat([pl.read_parquet(f'norm_test_s{k}.parquet',columns=['entity_id','business_name']) for k in (2,3)]).rename({'entity_id':'s23_id','business_name':'qname'})
with pl.Config(tbl_rows=25,fmt_str_lengths=45,tbl_cols=8):
    print(x.join(q,on='s23_id').join(s1.select(pl.col('entity_id').alias('s1_id'),'business_name'),on='s1_id').select('p2','business_name','qname','a1').sample(20,seed=2))
