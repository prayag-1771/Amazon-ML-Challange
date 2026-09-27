import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
VAR={'US':None,'India':None}
v=pl.read_parquet('swap_v.parquet')
s1=pl.read_parquet('norm_train_s1.parquet',columns=['country','name_core'])
f=s1.select('country',pl.col('name_core').str.split(' ').list.unique()).explode('name_core').group_by('country','name_core').len('f').join(s1.group_by('country').len('n'),on='country').with_columns((pl.col('f')/pl.col('n')*1000).alias('f')).drop('n')
v=v.join(f.rename({'name_core':'at','f':'af'}),on=['country','at'],how='left').join(f.rename({'name_core':'dt','f':'df'}),on=['country','dt'],how='left').fill_null(0)
# top swapped-in tokens: in/out ratio in val
g=v.group_by('country','at').agg(pl.len().alias('nin'),pl.col('lab').mean().round(3),pl.col('af').first().round(2)).sort('nin',descending=True)
print(g.filter(pl.col('country')=='US').head(25)); print(g.filter(pl.col('country')=='India').head(15))
w=v.filter((pl.col('sim')<.5)&(pl.col('af')>=.2)&(pl.col('at').str.len_chars()>=4))
top=g.group_by('country').head(8).select('country','at').with_columns(pl.lit(True).alias('topv'))
w=w.join(top,on=['country','at'],how='left').fill_null(False)
print(w.group_by('country','topv').agg(pl.len(),pl.col('lab').mean().round(3)))
print(w.filter(~pl.col('topv')&(pl.col('country')=='US')).sample(25,seed=2).select('n1','nq','lab','p2'))
