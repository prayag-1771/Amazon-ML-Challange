import polars as pl
t=pl.read_parquet('frtok.parquet')
x=t.filter((pl.col('bd').is_in(['low','lowmid']))&pl.col('aeq')&~pl.col('neq')&(pl.col('addr_empty')==0))
x=x.with_columns(pl.col('n_out').list.len().alias('no'),pl.col('n_in').list.len().alias('ni'))
with pl.Config(tbl_rows=30,fmt_str_lengths=45,tbl_cols=10):
    print(len(x)); print(x.group_by('no','ni').len().sort('len',descending=True).head(10))
    print(x.sample(25,seed=7).select('p','p2','n1','nq','l1','lq'))
