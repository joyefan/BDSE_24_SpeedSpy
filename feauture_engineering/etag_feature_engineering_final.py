import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as fn
import findspark
findspark.init('/usr/local/spark')

# 
spark = SparkSession\
        .builder\
        .master("yarn")\
        .config("spark.driver.memory", "16g")\
        .config("spark.driver.maxResultSize", "12g")\
        .config('spark.executor.instances','16')\
        .config('spark.executor.memory','9G')\
        .appName("feature engineering")\
        .getOrCreate()


# Check Spark Session
spark

spark.sparkContext.appName
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", True)


# ------------------------
# Load csv from HDFS
import pyspark.pandas as ps
ps.set_option("compute.default_index_type", "distributed")


# ------------------------
# read csv or parquet
# df = pd.read_csv('2020.csv')
df = ps.read_parquet(f'hdfs://nncluster:/data/etag/m05/parquet/')

# drop partition columns
df = df.drop(["month"],axis = 1)
print(f"Load CSV OK!!! , Shape{df.shape}")

# read holiday csv
df_holiday = ps.read_csv('hdfs://nncluster:/data/etag/m05/2015_2021_holiday.csv')
print(f"Load CSV OK!!! , Shape{df_holiday.shape}")

# add date string
df["Date_Start"] = ps.to_datetime(df["Time"],format = "%Y/%m/%d").dt.strftime('%Y%m%d')

# remane some station
df.loc[(df['S-Station']=='01F0509S'),'S-Station'] = '01F0511S'
df.loc[(df['S-Station']=='01F0509N'),'S-Station'] = '01F0511N'
df.loc[(df['E-Station']=='01F0509S'),'E-Station'] = '01F0511S'
df.loc[(df['E-Station']=='01F0509N'),'E-Station'] = '01F0511N'

# add Segment feature
df["Segment"] = df["S-Station"]+"-"+df["E-Station"]

# delete diff direction and road
def filters(x):
    if x[:2] != x[9:11]:
        return 0
    elif x[7] != x[-1]:
        return 0
    else:
        return 1

# keep wanted rows
df['Keep'] = df["Segment"].apply(lambda x : filters(x))
df = df[df['Keep']==1]

# add Direction feature
df["Direction"] = df["S-Station"].apply(lambda x: 0 if x[-1] == "S" else 1)

# add hour feature
df["Hour"] = df["Time"].apply(lambda x: int(x[-5:-3]))

# set speed = 0 to na and drop it
df.loc[df["Speed"]==0,"Speed"] = np.nan
df = df.dropna(subset="Speed")

# merge every hour data then sum count and avg speeds
df = df.groupby(['Hour','Date_Start','Segment', 'Direction','V-Type']).agg({"Count":"sum","Speed":"mean"}).reset_index()

# add year month weekday feature
date = ps.to_datetime(df["Date_Start"],format = "%Y%m%d")

df["Year"] = date.dt.strftime("%Y").astype(int)
df["Month"] = date.dt.strftime("%m").astype(int)
df["Wday"] = date.dt.strftime("%w").astype(int)

# add segment category feature
df["Seg_cat"] = df["Segment"]
df["Seg_cat"] = df["Seg_cat"].astype("category").cat.codes.astype(int)


# modify holiday dataframe and merge
df_holiday["Date Start"] = df_holiday["Date Start"].astype(str)
df_holiday = df_holiday.drop(["_c0"],axis=1)
df_holiday = df_holiday.rename(columns={'Date Start':'Date_Start'})
df = df.merge(df_holiday,how = "left")

# fill na in holiday feature
con = (df["Holiday"].isnull()) & (df["Wday"].isin([0,6]))
df.loc[con, "Holiday"] = 1

df["Holiday"] = df["Holiday"].fillna(0)

# change holiday dtype
df["Holiday"] = df["Holiday"].astype(int)

# sort data by date & segment
df = df.sort_values(["Date_Start","Seg_cat","Hour"])

print("="*50)
print("Feature Engineering Finished !!!")
print("Final DataFrame Shape",df.shape)

# output to parquet
df.to_parquet("hdfs://nncluster:/user/joyefan/2020_AFE_0521")

df = ps.read_parquet("hdfs://nncluster:/user/joyefan/2020_AFE_0521")

df.shape
type(df)

pdf = df.to_pandas()
pdf.shape
type(pdf)

pdf.to_csv('2020_AFE_0521.csv')