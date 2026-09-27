import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_width_chars(250)
c=['entity_id','country','name_core','addr_core','nums','addr_empty']
s1=pl.read_parquet('norm_train_s1.parquet',columns=c).filter(is_valid_expr('entity_id')).rename({'entity_id':'s1_id','name_core':'n1','addr_core':'a1','nums':'u1','addr_empty':'e1'})
q=pl.concat([pl.read_parquet(f'norm_train_s{k}.parquet',columns=c[:1]+c[2:]) for k in (2,3)]).rename({'entity_id':'s23_id','name_core':'nq','addr_core':'aq','nums':'uq','addr_empty':'eq'})
P=pl.read_parquet('pred15_v.parquet')
gt=load_ground_truth().join(s1.select('s1_id','country'),on='s1_id')
v=pl.read_parquet('valid_scores_stage2.parquet')
m=gt.join(P,on=['s1_id','s23_id'],how='anti').join(v.select('s1_id','s23_id','p2'),on=['s1_id','s23_id']).filter(pl.col('p2')>=.1)
# what is the top1 for those queries?
top=v.sort('p2',descending=True).unique('s23_id',keep='first').select('s23_id',pl.col('s1_id').alias('ts1'),pl.col('p2').alias('tp2'))
m=m.join(top,on='s23_id').with_columns((pl.col('ts1')==pl.col('s1_id')).alias('istop'))
m=m.join(s1.drop('country'),on='s1_id').join(q,on='s23_id')
f=lambda c: pl.col(c).fill_null('').str.split(' ').list.first()
m=m.with_columns((pl.col('n1')==pl.col('nq')).alias('neq'),(f('u1')==f('uq')).alias('ueq'))
print(m.group_by('country','istop','eq','neq','ueq').agg(pl.len(),pl.col('p2').median().round(2)).sort('len',descending=True).head(20))
print(m.filter((pl.col('country')=='US')&pl.col('istop')&(pl.col('eq')==0)).sample(30,seed=3).select('nq','n1','aq','a1',pl.col('p2').round(2)))
