import sys; sys.path.insert(0, ".")
import polars as pl
from src.io_utils import load_ground_truth
W="../work/"
q=pl.concat([pl.read_parquet(W+f"norm_train_s{i}.parquet") for i in (2,3)]).rename({"entity_id":"s23_id"})
print(q.columns)
e=q.filter(pl.col("addr_empty")==1)
print("raw address values for empty:", e.group_by(pl.col("business_address").fill_null("<null>")).len().sort("len",descending=True).head(8).rows())
print("state set", e.select((pl.col("state").fill_null("")!="").mean()).item())
gt=load_ground_truth()
s1=pl.read_parquet(W+"norm_train_s1.parquet").rename({"entity_id":"s1_id"})
x=e.join(gt,on="s23_id").join(s1.select("s1_id",pl.col("business_name").alias("bn1"),pl.col("name_core").alias("n1"),pl.col("legal").alias("l1"),pl.col("country").alias("c1")),on="s1_id")
nc=s1.group_by("country","name_core").len("k")
x=x.join(nc,left_on=["c1","n1"],right_on=["country","name_core"],how="left")
print("same-name S1 count for true S1 (empty queries):", x.select([pl.col("k").quantile(t).alias(str(t)) for t in (.25,.5,.75,.9)]).rows())
print("raw name equal", (x["business_name"]==x["bn1"]).mean(), "core equal", (x["name_core"]==x["n1"]).mean(), "legal equal", (x["legal"].fill_null("")==x["l1"].fill_null("")).mean())
for r in x.filter(pl.col("k")>5).sample(15,seed=1).select("business_name","bn1","k").rows(): print("  ",r)
