import polars as pl
W = '../work/'
s1 = pl.read_parquet(W + 'norm_test_s1.parquet', columns=['entity_id', 'business_name', 'business_address', 'country'])
q = pl.concat([pl.read_parquet(W + f'norm_test_s{k}.parquet', columns=['entity_id', 'business_name', 'business_address']) for k in (2, 3)])
sc = pl.read_parquet(W + 'test_scores_lgb_v5cf.parquet').join(s1.rename({'entity_id': 's1_id'}), on='s1_id').filter(pl.col('country') == 'France')
top = sc.sort('p', descending=True).group_by('s23_id', maintain_order=True).head(2)
unc = top.group_by('s23_id').agg(pl.col('p').max().alias('pmax')).filter((pl.col('pmax') > 0.1) & (pl.col('pmax') < 0.75)).sample(15, seed=3)
d = top.join(unc, on='s23_id').join(q.rename({'entity_id': 's23_id', 'business_name': 'qn', 'business_address': 'qa'}), on='s23_id').sort(['s23_id', 'p'], descending=[False, True])
for r in d.iter_rows(named=True):
    print(f"{r['s23_id']}  p={r['p']:.3f}\n   Q : {r['qn']} | {r['qa']}\n   S1: {r['business_name']} | {r['business_address']}")
