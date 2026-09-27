exec(open('struct16.py').read().split("bins=")[0])
bins=[.1,.5,.75,.9,.95,.98,.99]
f=lambda d:d.filter((pl.col('country')=='US')&(pl.col('addr_empty')==0)&pl.col('neq')&pl.col('numeq')&pl.col('aeq')&~pl.col('leq')).with_columns(pl.col('p2').cut(bins,left_closed=True).alias('band'))
nV=nv.filter(pl.col('country')=='US')['len'][0]; nT=nt.filter(pl.col('country')=='US')['len'][0]
A=f(v).group_by('band').agg(pl.len().alias('nv'),pl.col('lab').mean().round(3).alias('precV')).with_columns((pl.col('nv')/nV*1e4).round(1).alias('cv'))
B=f(t).group_by('band').agg(pl.len().alias('nt')).with_columns((pl.col('nt')/nT*1e4).round(1).alias('ct'))
with pl.Config(tbl_rows=20): print(B.join(A,on='band',how='full',coalesce=True).sort('band'))
