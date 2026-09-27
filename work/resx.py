import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.config import is_valid_expr
from src.io_utils import load_ground_truth
from src.metric import macro_f05
T={'US':.75,'India':.8}
s1=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id','country']).filter(is_valid_expr('entity_id')).rename({'entity_id':'s1_id'})
gt=load_ground_truth().join(s1.select('s1_id'),on='s1_id',how='semi')
v=pl.read_parquet('valid_scores_stage2.parquet').join(s1,on='s1_id')
top=v.sort('p2',descending=True).unique('s23_id',keep='first')
base=top.filter(pl.col('p2')>=pl.col('country').replace_strict(T)).select('s1_id','s23_id')
R=pl.read_parquet('v10/rescue_valid.parquet'); print(R.columns)
rt=R.sort('pr',descending=True).unique('s23_id',keep='first')
r=rt.filter(pl.col('vs1')&(pl.col('pr')>=.9)).join(v.select('s23_id').unique(),on='s23_id',how='anti')
casc=pl.read_parquet('cascade_valid_p001.parquet').select('s23_id').unique()
a=pl.read_parquet('v10/ac2_valid_mid.parquet').sort('pr',descending=True).unique('q_row',keep='first').filter(pl.col('vs1')).join(casc,on='s23_id',how='anti').join(base,on='s23_id',how='anti').filter(pl.col('pr')>=.9)
b15=pl.concat([base,r.select('s1_id','s23_id'),a.select('s1_id','s23_id')])
m0=macro_f05(s1['s1_id'],b15,gt,by=s1); print('v15',{k:round(x,5) for k,x in m0.items()})
# extension: queries in cascade, with no accepted pair
x=rt.filter(pl.col('vs1')).join(v.select('s23_id').unique(),on='s23_id',how='semi').join(b15,on='s23_id',how='anti')
x=x.join(top.select('s23_id',pl.col('s1_id').alias('ctop'),pl.col('p2').alias('cp2')),on='s23_id',how='left')
g1=gt.with_columns(pl.lit(1).alias('lab'))
x=x.join(g1,on=['s1_id','s23_id'],how='left').with_columns(pl.col('lab').fill_null(0),(pl.col('s1_id')==pl.col('ctop')).alias('agree'))
for t in (.8,.9,.95,.97):
  for ag in (None,True,False):
    add=x.filter(pl.col('pr')>=t)
    if ag is not None: add=add.filter(pl.col('agree')==ag)
    m=macro_f05(s1['s1_id'],pl.concat([b15,add.select('s1_id','s23_id')]),gt,by=s1)
    print(t,ag,len(add),round(add['lab'].mean(),3),round(m['f05']-m0['f05'],5),round(m['f05_US']-m0['f05_US'],5),round(m['f05_India']-m0['f05_India'],5))
