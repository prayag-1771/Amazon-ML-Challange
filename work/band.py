import polars as pl
W="../work/"
s = pl.read_parquet(W+"test_scores_lgb_v6k.parquet")
print(len(s), s.filter((pl.col("p")>0.003)&(pl.col("p")<0.997)).height, s.filter((pl.col("p")>0.01)&(pl.col("p")<0.99)).height)
