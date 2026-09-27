import polars as pl
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist
def sim(d):
    s=cpdist(d['o'].to_list(),d['i'].to_list(),scorer=Levenshtein.normalized_similarity,workers=-1)
    d=d.with_columns(pl.Series('ls',s))
    pre=pl.col('o').str.starts_with(pl.col('i'))|pl.col('i').str.starts_with(pl.col('o'))
    nd=(pl.col('i')==pl.lit('ndeg')+pl.col('o'))|(pl.col('o')==pl.lit('ndeg')+pl.col('i'))
    return d.with_columns(((pl.col('ls')>=.6)|pre|nd).alias('similar'))
t=sim(pl.read_parquet('asw_t.parquet')); v=sim(pl.read_parquet('asw_v.parquet'))
with pl.Config(tbl_rows=60,tbl_cols=10,fmt_str_lengths=30):
    print(v.filter(pl.col('samedig')).group_by('country','similar',pl.col('p2')>=.75).agg(pl.len(),pl.col('lab').mean()).sort('country','similar','p2'))
    f=t.filter((pl.col('country')=='France')&pl.col('samedig'))
    print(f.group_by('similar','band').agg(pl.len(),pl.col('acc').mean()).sort('similar','band'))
    print(f.filter(pl.col('similar')&(pl.col('acc')==0)).select('p','p2','n1','o','i','a1').sample(25,seed=1))
    print(f.filter(~pl.col('similar')&(pl.col('acc')==0)).select('p','p2','n1','o','i','a1').sample(15,seed=1))
