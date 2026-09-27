import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
t=pl.read_parquet('swap_t.parquet').filter(pl.col('country')=='France')
s1=pl.read_parquet('norm_test_s1.parquet',columns=['country','name_core']).filter(pl.col('country')=='France')
n=s1.height
f=s1.select(pl.col('name_core').str.split(' ').list.unique()).explode('name_core').group_by('name_core').len('f').with_columns(pl.col('f')/n*1000).rename({'name_core':'tok'})
t=t.join(f.rename({'tok':'at','f':'af'}),on='at',how='left').join(f.rename({'tok':'dt','f':'df'}),on='dt',how='left').fill_null(0)
t=t.with_columns((pl.col('af')>=.2).alias('areal'),(pl.col('df')>=.2).alias('dreal'),(pl.col('p2')>=.75).alias('acc'))
print(t.group_by('sb','areal','dreal').agg(pl.len(),pl.col('acc').mean().round(3),pl.col('p2').median().round(3)).sort('sb','areal','dreal'))
# top swapped-in tokens among unaccepted low sim
x=t.filter(pl.col('sim')<=.3)
g=x.group_by('at').agg(pl.len(),pl.col('acc').mean().round(2),pl.col('af').first().round(2)).sort('len',descending=True)
print(g.head(40))
