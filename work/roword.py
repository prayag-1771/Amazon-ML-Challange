import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl, numpy as np
from src.io_utils import load_ground_truth
gt=load_ground_truth()
s1=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id']).with_row_index('r1').rename({'entity_id':'s1_id'})
s2=pl.read_parquet('norm_train_s2.parquet',columns=['entity_id']).with_row_index('r2').rename({'entity_id':'s23_id'})
x=gt.join(s1,on='s1_id').join(s2,on='s23_id')
print(len(x), np.corrcoef(x['r1'].to_numpy().astype(float),x['r2'].to_numpy().astype(float))[0,1])
i1=x['s1_id'].str.slice(3).cast(pl.Int64).to_numpy(); i2=x['s23_id'].str.slice(3).cast(pl.Int64).to_numpy()
print('id corr',np.corrcoef(i1,i2)[0,1])
