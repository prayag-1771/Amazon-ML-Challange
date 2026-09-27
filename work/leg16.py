exec(open('struct16.py').read().split("bins=")[0])
for nm,d in (('val',v),('test',t)):
    x=d.filter((pl.col('country')=='US')&(pl.col('addr_empty')==0)&pl.col('neq')&~pl.col('leq')&(pl.col('p2')>=.5))
    x=x.with_columns(pl.col('p2').cut([.75,.9,.95,.98,.99],left_closed=True).alias('band'),pl.col('aeq'))
    agg=[pl.len()]+([pl.col('lab').mean().alias('prec')] if nm=='val' else [])
    with pl.Config(tbl_rows=30): print(nm, x.group_by('band','aeq').agg(agg).sort('aeq','band'))
t.filter((pl.col('country')=='US')&(pl.col('addr_empty')==0)&pl.col('neq')&pl.col('aeq')&~pl.col('leq')&pl.col('p2').is_between(.98,.99)).write_parquet('leg16_t.parquet')
v.filter((pl.col('country')=='US')&(pl.col('addr_empty')==0)&pl.col('neq')&~pl.col('leq')&(pl.col('p2')>=.75)).write_parquet('leg16_v.parquet')
with pl.Config(tbl_rows=40,fmt_str_lengths=40,tbl_cols=10):
    print(pl.read_parquet('leg16_t.parquet').group_by('l1','lq').len().sort('len',descending=True).head(15))
    print(pl.read_parquet('leg16_v.parquet').group_by('l1','lq').agg(pl.len(),pl.col('lab').mean()).sort('len',descending=True).head(15))
