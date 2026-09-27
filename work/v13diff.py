import polars as pl
def rd(p,c):
    return pl.read_csv(p,separator="\t",quote_char=None,schema_overrides={c:pl.Utf8}).with_columns(pl.col(c).fill_null("").str.split(",")).explode(c).filter(pl.col(c)!="").rename({c:"s23_id","source1_entity_id":"s1_id"})
m13=rd("output/matching_results.tsv","matched_entity_ids"); m12=rd("work/sub12_cascade/matching_results.tsv","matched_entity_ids")
c13=rd("output/candidate_pairs.tsv","candidate_entity_ids")
res=pl.read_parquet("work/rescue_test_accept.parquet").select("s1_id","s23_id")
print("m13",len(m13),"m12",len(m12),"added",len(m13.join(m12,on=["s1_id","s23_id"],how="anti")),"removed",len(m12.join(m13,on=["s1_id","s23_id"],how="anti")))
print("added==rescue", m13.join(m12,on=["s1_id","s23_id"],how="anti").join(res,on=["s1_id","s23_id"],how="anti").height==0)
print("matches not in cand", m13.join(c13,on=["s1_id","s23_id"],how="anti").height, "cand",len(c13), "dup s23 in matches", m13["s23_id"].is_duplicated().sum())
