import polars as pl
t=pl.read_parquet('test_scores_stage2_v15.parquet')
s1=pl.read_parquet('norm_test_s1.parquet',columns=['entity_id','country','name_core','nums','legal','addr_core']).rename({'entity_id':'s1_id','name_core':'n1','nums':'u1','legal':'l1','addr_core':'a1'})
q=pl.concat([pl.read_parquet(f'norm_test_s{k}.parquet',columns=['entity_id','addr_empty','name_core','nums','legal','addr_core']) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','nums':'uq','legal':'lq','addr_core':'aq'})
top=t.sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id')
a=pl.col('u1').str.split(' ').list.first(); b=pl.col('uq').str.split(' ').list.first()
top=top.with_columns((pl.col('n1')==pl.col('nq')).alias('neq'),(a==b).fill_null(False).alias('numeq'),(pl.col('a1')==pl.col('aq')).alias('aeq'),(pl.col('l1').fill_null('')==pl.col('lq').fill_null('')).alias('leq'),(pl.col('p2')*10).floor().alias('b'))
with pl.Config(tbl_rows=60, tbl_cols=20):
    print(top.filter(pl.col('addr_empty')==0).group_by('country','b').len().pivot(on='country',index='b',values='len').sort('b'))
    mid=top.filter((pl.col('addr_empty')==0)&pl.col('p2').is_between(.3,.75))
    print(mid.group_by('country','neq','numeq').len().sort('country','neq','numeq'))
    print(mid.filter(pl.col('country')=='France').sample(30,seed=1).select('p2','n1','nq','u1','uq','l1','lq'))
top.write_parquet('top15_test.parquet')
