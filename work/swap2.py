import polars as pl
from rapidfuzz.distance import Levenshtein
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
tok=lambda x: pl.col(x).fill_null('').str.split(' ').list.eval(pl.element().filter(pl.element()!='')).list.unique()
def prep(f):
    c=pl.read_parquet(f).filter(pl.col('numeq')&pl.col('aeq')&(pl.col('addr_empty')==0)&(pl.col('a1cnt')==1)&~pl.col('neq'))
    c=c.with_columns(tok('n1').alias('t1'),tok('nq').alias('tq'))
    c=c.with_columns(pl.col('tq').list.set_difference('t1').alias('A'),pl.col('t1').list.set_difference('tq').alias('D'))
    c=c.filter((pl.col('A').list.len()==1)&(pl.col('D').list.len()==1)).with_columns(pl.col('A').list.first().alias('at'),pl.col('D').list.first().alias('dt'))
    c=c.with_columns(pl.struct('at','dt').map_elements(lambda r: Levenshtein.normalized_similarity(r['at'],r['dt']),return_dtype=pl.Float64).alias('sim'))
    return c.with_columns(pl.col('sim').cut([.3,.5,.7,.85]).alias('sb'))
v=prep('aeq_v.parquet'); t=prep('aeq_t.parquet')
print(v.group_by('country','sb').agg(pl.len(),pl.col('lab').mean().round(3)).sort('country','sb'))
print(t.group_by('country','sb').agg(pl.len(),(pl.col('p2')>=.75).mean().round(3).alias('acc'),pl.col('p2').median().round(3)).sort('country','sb'))
print(v.filter((pl.col('country')=='US')&(pl.col('sim')<.3)).sample(20,seed=1).select('dt','at','lab','p2'))
t.write_parquet('swap_t.parquet'); v.write_parquet('swap_v.parquet')
