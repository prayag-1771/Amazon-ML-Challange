import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
T={'US':.75,'India':.8}
s1=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id','country']).filter(is_valid_expr('entity_id')).rename({'entity_id':'s1_id'})
gt=load_ground_truth().with_columns(pl.lit(1).alias('lab'))
v=pl.read_parquet('valid_scores_stage2.parquet').join(s1,on='s1_id')
top=v.sort('p2',descending=True).unique('s23_id',keep='first')
acc=top.filter(pl.col('p2')>=pl.col('country').replace_strict(T)).select('s1_id','s23_id')
cnt=acc.group_by('s1_id').agg(pl.col('s23_id').str.starts_with('S2').sum().alias('a2'),pl.col('s23_id').str.starts_with('S3').sum().alias('a3'))
R=pl.read_parquet('v10/rescue_valid.parquet').filter(pl.col('vs1')).join(acc,on='s23_id',how='anti')   # rescue candidates for unaccepted empty queries
R=R.join(cnt,on='s1_id',how='left').fill_null(0).with_columns(pl.col('s23_id').str.slice(0,2).alias('src'))
R=R.with_columns(pl.when(pl.col('src')=='S2').then(pl.col('a2')).otherwise(pl.col('a3')).alias('same_src'),pl.when(pl.col('src')=='S2').then(pl.col('a3')).otherwise(pl.col('a2')).alias('oth_src'))
R=R.with_columns(pl.col('label').cast(pl.Int8))
print(R.group_by('label').agg(pl.len(),(pl.col('same_src')==0).mean().alias('same0'),pl.col('same_src').mean(),pl.col('oth_src').mean(),((pl.col('same_src')+pl.col('oth_src'))==0).mean().alias('all0')))
# among queries with a true candidate: in exact-name ties, does "same_src==0" pick the true one?
h=R.filter(pl.col('label').max().over('q_row')==1).filter(pl.col('eq')==1)
h=h.with_columns(pl.len().over('q_row').alias('nt')).filter(pl.col('nt')>=2)
print('tied exact-name queries', h['q_row'].n_unique())
print(h.group_by("label").agg((pl.col("same_src")==0).mean().alias("same0"),pl.col('same_src').mean(),pl.col('oth_src').mean(),pl.col('s1_empty').mean()))
