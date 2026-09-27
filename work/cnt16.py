import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.io_utils import load_ground_truth
gt=load_ground_truth()
s1=pl.read_parquet('norm_train_s1.parquet',columns=['entity_id','country']).rename({'entity_id':'s1_id'})
c=s1.join(gt.group_by('s1_id').agg(pl.col('s23_id').str.starts_with('S2').sum().alias('a2'),pl.col('s23_id').str.starts_with('S3').sum().alias('a3')),on='s1_id',how='left').fill_null(0)
with pl.Config(tbl_rows=40): print(c.group_by('a2','a3').len().with_columns((pl.col('len')/pl.col('len').sum()).round(4).alias('f')).sort('len',descending=True).head(20))
