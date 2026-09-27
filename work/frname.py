import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
pl.Config.set_tbl_rows(60); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_width_chars(250)
def load(split):
    c=['entity_id','country','name_core','addr_core','nums']
    s1=pl.read_parquet(f'norm_{split}_s1.parquet',columns=c).rename({'entity_id':'s1_id','name_core':'n1','addr_core':'a1','nums':'u1'})
    q=pl.concat([pl.read_parquet(f'norm_{split}_s{k}.parquet',columns=c[:1]+c[2:]) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','addr_core':'aq','nums':'uq'})
    return s1.join(s1.group_by('country','n1').len('nc'),on=['country','n1']),q
srt=lambda c: pl.col(c).fill_null("").str.split(" ").list.sort().list.join(" ")
def prep(t):
    t=t.filter((pl.col("aq")!="")&(pl.col("n1")==pl.col("nq"))&(srt("a1")!=srt("aq")))
    f=lambda c: pl.col(c).fill_null("").str.split(" ").list.first()
    return t.with_columns((f("u1")==f("uq")).alias("n1eq"),(pl.col("nc")==1).alias("uniq"))
s1,q=load('test')
t=pl.read_parquet('test_scores_stage2_v15.parquet').sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id')
t=prep(t)
print(t.group_by('country','uniq','n1eq').agg(pl.len(),(pl.col('p2')>=.75).mean().round(3).alias('hi'),pl.col('p2').median().round(2).alias('med')).sort('country','uniq','n1eq'))
s1,q=load('train'); s1=s1.filter(is_valid_expr('s1_id'))
v=prep(pl.read_parquet('valid_scores_stage2.parquet').sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id'))
print(v.group_by('country','uniq','n1eq').agg(pl.len(),pl.col('label').mean().round(3).alias('lab'),(pl.col('p2')>=.75).mean().round(3).alias('hi')).sort('country','uniq','n1eq'))
print(t.filter((pl.col('country')=='France')&pl.col('uniq')&(pl.col('p2')<.75)).sample(30,seed=5).select('nq','aq','a1',pl.col('p2').round(2)))
