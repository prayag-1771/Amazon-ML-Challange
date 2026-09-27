import polars as pl
pl.Config.set_tbl_rows(50); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(260)
t=pl.read_parquet('frtok.parquet').filter(pl.col('p2').is_between(.1,.75))
t=t.with_columns(pl.col('n_out').list.len().alias('no'),pl.col('n_in').list.len().alias('ni'),pl.col('a_out').list.len().alias('ao'),pl.col('a_in').list.len().alias('ai'))
g=t.group_by('addr_empty','neq','numeq','aeq','leq').agg(pl.len(),pl.col('p2').mean().round(2)).sort('len',descending=True)
print(g.head(25))
print(t.sample(40,seed=3).select('nq','n1','aq','a1','uq','u1','p2'))
