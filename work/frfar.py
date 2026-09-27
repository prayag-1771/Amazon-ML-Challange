import polars as pl
t=pl.read_parquet('frtok.parquet')
a=pl.col('u1').str.split(' ').list.first().cast(pl.Int64,strict=False); b=pl.col('uq').str.split(' ').list.first().cast(pl.Int64,strict=False)
t=t.with_columns((b-a).alias('sh'))
x=t.filter((pl.col('n_out').list.len()==0)&(pl.col('n_in').list.len()==0)&pl.col('sh').is_not_null()&(pl.col('sh')!=0))
with pl.Config(tbl_rows=40,fmt_str_lengths=40,tbl_cols=10):
    print(x.group_by('bd').agg(pl.len(),(pl.col('sh')>0).mean().alias('up'),pl.col('sh').abs().median().alias('medabs')))
    print(x.group_by(pl.col('sh').clip(-30,30)).agg(pl.len()).sort('len',descending=True).head(20))
    print(x.filter(pl.col('bd')=='mid').select('p','p2','n1','a1','aq').sample(12,seed=5))
