import polars as pl
def load(p,c):
    return pl.read_csv(p,separator='\t').with_columns(pl.col(c).str.split(',')).explode(c).drop_nulls().rename({c:'s23_id','source1_entity_id':'s1_id'})
m=load('sub15_indiaaddr/matching_results.tsv','matched_entity_ids').with_columns(pl.lit(True).alias('acc'))
top=pl.read_parquet('top15_test.parquet').join(m,on=['s1_id','s23_id'],how='left').with_columns(pl.col('acc').fill_null(False))
mid=top.filter((pl.col('addr_empty')==0)&pl.col('p2').is_between(.3,.75))
print(mid.group_by('country','acc').len().sort('country','acc'))
t1=pl.col('n1').str.split(' '); tq=pl.col('nq').str.split(' ')
x=mid.filter((pl.col('country')=='France')&~pl.col('acc')&pl.col('numeq')).with_columns(tq.list.set_difference(t1).alias('add'),t1.list.set_difference(tq).alias('drop'))
x=x.with_columns(pl.col('add').list.len().alias('na'),pl.col('drop').list.len().alias('nd'))
print(x.group_by('na','nd').len().sort('len',descending=True).head(8))
with pl.Config(tbl_rows=30):
    print(x.filter((pl.col('na')==1)&(pl.col('nd')==1)).with_columns(pl.col('add').list.first().alias('a'),pl.col('drop').list.first().alias('d')).group_by('a').len().sort('len',descending=True).head(15))
    print(x.filter((pl.col('na')==1)&(pl.col('nd')==0)).with_columns(pl.col('add').list.first().alias('a')).group_by('a').len().sort('len',descending=True).head(10))
    print(x.filter((pl.col('na')==0)&(pl.col('nd')==1)).with_columns(pl.col('drop').list.first().alias('d')).group_by('d').len().sort('len',descending=True).head(10))
z=x.filter((pl.col('na')==0)&(pl.col('nd')==0))
print(z.group_by('leq','aeq').len())
with pl.Config(tbl_rows=25, fmt_str_lengths=40): print(z.sample(25,seed=2).select('p2','n1','nq','l1','lq','a1','aq'))
