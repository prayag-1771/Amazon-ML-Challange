import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
acc=pl.read_csv('../output/matching_results.tsv',separator='\t',infer_schema_length=0).with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').drop_nulls().select(pl.col('source1_entity_id').alias('s1_id'),pl.col('matched_entity_ids').alias('s23_id')).with_columns(pl.lit(True).alias('fin'))
t=pl.read_parquet('swap_t.parquet').filter(pl.col('country')=='France').join(acc,on=['s1_id','s23_id'],how='left').with_columns(pl.col('fin').fill_null(False))
s1=pl.read_parquet('norm_test_s1.parquet',columns=['country','name_core']).filter(pl.col('country')=='France'); n=s1.height
f=s1.select(pl.col('name_core').str.split(' ').list.unique()).explode('name_core').group_by('name_core').len('f').with_columns(pl.col('f')/n*1000)
t=t.join(f.rename({'name_core':'at','f':'af'}),on='at',how='left').join(f.rename({'name_core':'dt','f':'df'}),on='dt',how='left').fill_null(0)
print(t.group_by('sb').agg(pl.len(),pl.col('fin').mean().round(3)).sort('sb'))
g=t.filter(pl.col('sim')<=.3).group_by('at').agg(pl.len(),pl.col('fin').mean().round(2),pl.col('af').first().round(2)).sort('len',descending=True)
print(g.head(30))
t.write_parquet('swap_fr_fin.parquet')
