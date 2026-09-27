import polars as pl
pl.Config.set_tbl_rows(40)
t=pl.read_parquet('frtok.parquet')
tt=pl.read_parquet('top15_test.parquet')
print(tt.columns)
b=tt.with_columns(pl.col('p2').cut([.05,.1,.2,.3,.5,.6,.7,.75,.8,.9,.97]).alias('bin')).group_by('country','bin').len().pivot(on='country',index='bin',values='len').sort('bin')
tot=tt.group_by('country').len()
print(b); print(tot)
