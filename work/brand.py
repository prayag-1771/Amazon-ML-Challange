import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250)
W = '../work/'
cols = ['entity_id', 'business_name', 'business_address', 'country', 'name_core', 'name_alt', 'addr_core']
s1 = pl.read_parquet(W + 'norm_train_s1.parquet', columns=cols)
q = pl.concat([pl.read_parquet(W + f'norm_train_s{k}.parquet', columns=cols[:3] + ['name_core', 'name_alt', 'addr_core']) for k in (2, 3)])
gt = pl.read_parquet(W + 'gt_pairs.parquet')
print(gt.columns)
gt = gt.rename({gt.columns[0]: 's1_id', gt.columns[1]: 's23_id'})
d = gt.join(s1.rename({c: c + '1' for c in cols if c != 'entity_id'}).rename({'entity_id': 's1_id'}), on='s1_id') \
      .join(q.rename({'entity_id': 's23_id'}), on='s23_id')
d = d.with_columns(pl.Series('nsim', cpdist(d['name_core1'].to_list(), d['name_core'].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
print(d.group_by('country1').agg((pl.col('nsim') < 40).mean().alias('low_name_sim'), pl.len()))
low = d.filter(pl.col('nsim') < 40)
print(low.sample(25, seed=1).select('business_name1', 'business_name', 'business_address1', 'business_address'))
# do other true queries of the same S1 carry the low-sim name somehow?
sib = d.select('s1_id', pl.col('business_name').alias('sib_name'), pl.col('s23_id').alias('sib_id'))
ex = low.sample(8, seed=2).select('s1_id', 's23_id', 'business_name1', 'business_name').join(sib, on='s1_id')
for r in ex.iter_rows(named=True):
    print(r['s1_id'], '|', r['business_name1'], '| Q:', r['business_name'], '| sib:', r['sib_name'])
