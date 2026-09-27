import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.stage2 import _tok
pl.Config.set_tbl_rows(80); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(250)
def load(split):
    c=['entity_id','country','name_core','addr_core']
    s1=pl.read_parquet(f'norm_{split}_s1.parquet',columns=c).rename({'entity_id':'s1_id','name_core':'n1','addr_core':'a1'})
    q=pl.concat([pl.read_parquet(f'norm_{split}_s{k}.parquet',columns=c[:1]+c[2:]) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','addr_core':'aq'})
    return s1,q
srt=lambda c: pl.col(c).fill_null("").str.split(" ").list.sort().list.join(" ")
def prep(t):
    t=t.filter((srt("a1")==srt("aq"))&(pl.col("aq")!="")&(pl.col("n1")!=pl.col("nq")))
    t=t.with_columns(_tok("n1").alias("t1"),_tok("nq").alias("tq"))
    t=t.with_columns(pl.col("tq").list.set_difference("t1").alias("A"),pl.col("t1").list.set_difference("tq").alias("D"))
    return t.with_columns(pl.col("A").list.len().clip(0,3).alias("na"),pl.col("D").list.len().clip(0,3).alias("nd"),
        (pl.col("t1").list.sort()==pl.col("tq").list.sort()).alias("perm"))
rows=[]
with open('sub15_indiaaddr/matching_results.tsv') as f:
    next(f)
    for l in f:
        a,b=l.rstrip('\n').split('\t')
        if b: rows+=[(a,x) for x in b.split(',')]
M=pl.DataFrame(rows,schema=['s1_id','s23_id'],orient='row').with_columns(pl.lit(1).alias('acc'))
s1,q=load('test')
t=pl.read_parquet('test_scores_stage2_v15.parquet').sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id')
t=prep(t).join(M,on=['s1_id','s23_id'],how='left').with_columns(pl.col('acc').fill_null(0))
T=t.group_by('country','perm','na','nd').agg(pl.len(),pl.col('acc').mean().round(3).alias('acc'),pl.col('p2').median().round(2).alias('p2m'))
s1,q=load('train'); s1=s1.filter(is_valid_expr('s1_id'))
v=pl.read_parquet('valid_scores_stage2.parquet').sort('p2',descending=True).unique('s23_id',keep='first').join(s1,on='s1_id').join(q,on='s23_id')
v=prep(v)
V=v.group_by('country','perm','na','nd').agg(pl.len().alias('vn'),pl.col('label').mean().round(3).alias('lab'),(pl.col('p2')>=.75).mean().round(3).alias('vacc'))
X=T.pivot(on='country',index=['perm','na','nd'],values=['len','acc']).join(V.filter(pl.col('country')=='US').drop('country'),on=['perm','na','nd'],how='left').sort('perm','na','nd')
print(X)
t.write_parquet('frcell_t.parquet')
