import polars as pl
pl.Config.set_tbl_rows(60)
tok=lambda x: pl.col(x).fill_null('').str.split(' ').list.eval(pl.element().filter(pl.element()!='')).list.unique()
def prep(f):
    c=pl.read_parquet(f).filter(pl.col('numeq')&pl.col('aeq')&(pl.col('addr_empty')==0)&(pl.col('a1cnt')==1)&~pl.col('neq'))
    c=c.with_columns(tok('n1').alias('t1'),tok('nq').alias('tq'))
    c=c.with_columns(pl.col('tq').list.set_difference('t1').list.len().alias('na'),pl.col('t1').list.set_difference('tq').list.len().alias('nd'),pl.col('tq').list.set_intersection('t1').list.len().alias('sh'))
    return c.with_columns(pl.col('p2').cut([.1,.5,.75]).alias('b'))
v=prep('aeq_v.parquet'); t=prep('aeq_t.parquet')
print(v.filter(pl.col('sh')>0).with_columns(pl.col('na').clip(0,2),pl.col('nd').clip(0,2)).group_by('country','na','nd').agg(pl.len(),pl.col('lab').mean().round(3),(pl.col('p2')>=.75).mean().round(3).alias('acc')).sort('country','na','nd'))
print(t.filter(pl.col('sh')>0).with_columns(pl.col('na').clip(0,2),pl.col('nd').clip(0,2)).group_by('country','na','nd').agg(pl.len(),(pl.col('p2')>=.75).mean().round(3).alias('acc')).sort('country','na','nd'))
