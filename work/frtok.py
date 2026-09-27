import polars as pl
t=pl.read_parquet('top15_test.parquet').filter(pl.col('country')=='France')
tok=lambda c: pl.col(c).fill_null('').str.split(' ').list.eval(pl.element().filter(pl.element()!=''))
t=t.with_columns(tok('n1').alias('tn1'),tok('nq').alias('tnq'),tok('a1').alias('ta1'),tok('aq').alias('taq'))
t=t.with_columns(pl.col('tn1').list.set_difference('tnq').alias('n_out'),pl.col('tnq').list.set_difference('tn1').alias('n_in'),
                 pl.col('ta1').list.set_difference('taq').alias('a_out'),pl.col('taq').list.set_difference('ta1').alias('a_in'))
t=t.with_columns(pl.when(pl.col('p2')>=.9).then(pl.lit('hi')).when(pl.col('p2')>=.3).then(pl.lit('mid')).when(pl.col('p2')>=.1).then(pl.lit('lowmid')).otherwise(pl.lit('low')).alias('bd'))
for side in ('n','a'):
    x=t.filter((pl.col(side+'_out').list.len()==1)&(pl.col(side+'_in').list.len()==1)).with_columns(pl.col(side+'_out').list.first().alias('o'),pl.col(side+'_in').list.first().alias('i'))
    g=x.group_by('o','i').agg(pl.len().alias('n'),*[(pl.col('bd')==b).sum().alias(b) for b in ('hi','mid','lowmid','low')]).sort('mid',descending=True)
    with pl.Config(tbl_rows=40): print(side,'swap', g.head(35))
x=t.filter(pl.col('bd')=='mid').with_columns(pl.col('n_out').list.len().alias('no'),pl.col('n_in').list.len().alias('ni'),pl.col('a_out').list.len().alias('ao'),pl.col('a_in').list.len().alias('ai'))
with pl.Config(tbl_rows=30): print(x.group_by('addr_empty','no','ni','ao','ai').len().sort('len',descending=True).head(25))
t.write_parquet('frtok.parquet')
