import sys; sys.path.insert(0,'../business_entity_resolution')
import polars as pl
from src.io_utils import load_ground_truth
for split in ('train','test'):
    s1=pl.read_parquet(f'norm_{split}_s1.parquet',columns=['entity_id','country'])
    q=pl.concat([pl.read_parquet(f'norm_{split}_s{k}.parquet',columns=['entity_id','country','addr_empty']) for k in (2,3)])
    cas=pl.read_parquet(f'cascade_{"valid" if split=="train" else "test"}_p001.parquet',columns=['s23_id']).unique() if split=='test' else None
    out=q.group_by('country').agg(pl.len().alias('nq'),pl.col('addr_empty').mean().alias('empty')).join(s1.group_by('country').len('ns1'),on='country')
    out=out.with_columns((pl.col('nq')/pl.col('ns1')).alias('q_per_s1'))
    if split=='train':
        gt=load_ground_truth().join(s1.rename({'entity_id':'s1_id'}),on='s1_id')
        out=out.join(gt.group_by('country').agg(pl.len().alias('ntrue')),on='country').with_columns((pl.col('ntrue')/pl.col('nq')).alias('matched_frac'),(pl.col('ntrue')/pl.col('ns1')).alias('true_per_s1'))
    else:
        out=out.join(q.join(cas.rename({'s23_id':'entity_id'}),on='entity_id',how='semi').group_by('country').len('ncand'),on='country').with_columns((pl.col('ncand')/pl.col('nq')).alias('cand_frac'))
        m=pl.read_csv('../output/matching_results.tsv',separator='\t').with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').drop_nulls()
        m=m.join(s1.rename({'entity_id':'source1_entity_id'}),on='source1_entity_id')
        out=out.join(m.group_by('country').len('nacc'),on='country').with_columns((pl.col('nacc')/pl.col('nq')).alias('acc_frac'),(pl.col('nacc')/pl.col('ns1')).alias('acc_per_s1'))
    with pl.Config(tbl_cols=20): print(split,out)
