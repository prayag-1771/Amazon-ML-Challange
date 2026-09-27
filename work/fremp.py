import polars as pl
pl.Config.set_tbl_rows(40)
q=pl.concat([pl.read_parquet(f'norm_test_s{k}.parquet',columns=['entity_id','country','addr_empty']) for k in (2,3)]).rename({'entity_id':'s23_id'}).filter(pl.col('addr_empty')==1)
c=pl.read_parquet('cascade_test_p001.parquet').select('s23_id').unique().with_columns(pl.lit(True).alias('hascand'))
t=pl.read_parquet('top15_test.parquet').select('s23_id','p2')
r=pl.read_parquet('rescue_test_accept.parquet').select('s23_id').with_columns(pl.lit(True).alias('resc'))
d=q.join(c,on='s23_id',how='left').join(t,on='s23_id',how='left').join(r,on='s23_id',how='left').fill_null(False)
d=d.with_columns(pl.when(~pl.col('hascand')).then(pl.lit('nocand')).when(pl.col('p2')>=.75).then(pl.lit('hi')).when(pl.col('p2')>=.3).then(pl.lit('mid')).otherwise(pl.lit('low')).alias('k'))
print(d.group_by('country','k').agg(pl.len(),pl.col('resc').sum()).with_columns((pl.col('len')/pl.col('len').sum().over('country')).round(3).alias('frac')).sort('country','k'))
