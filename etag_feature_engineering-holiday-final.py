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
# read csv
# df = pd.read_csv('2015_2021_holidays.csv')
df = ps.read_parquet(f'hdfs://nncluster:/data/etag/m05/m05_holidays.parquet/')
print(f"Load CSV , Shape{df.shape}")

# read holiday with order csv
df1 = ps.read_csv(f'hdfs://nncluster:/data/etag/m05/2015_2021_holidays_order.csv')
print(f"Load CSV , Shape{df1.shape}")

# add date string
df["Date_Start"] = ps.to_datetime(df["Time"],format = "%Y/%m/%d").dt.strftime('%Y%m%d')

# df1 drop useless columns
df1 = df1.drop(['_c0','Year'],axis=1)

# pick specific holiday
df1 = df1[df1['Holiday'].isin(['Lunar New Year','dragon','moon','New Year','national'])]
df1 = df1.rename(columns = {"Date Start":"Date_Start"})

# merge order df to main df
df = df.merge(df1,on='Date_Start')

# save a parquet(just in case)
df.to_parquet("hdfs://nncluster:/data/etag/m05/m05_holidays_0521_step1.parquet", partition_cols="year")
df = ps.read_parquet(f'hdfs://nncluster:/data/etag/m05/m05_holidays_0521_step1.parquet/')

# df2-6 preprocess
df2 = ps.read_csv(f'hdfs://nncluster:/data/etag/m05/m05_DaysPriorToDragon.csv')
df3 = ps.read_csv(f'hdfs://nncluster:/data/etag/m05/m05_DaysPriorToLunar.csv')
df4 = ps.read_csv(f'hdfs://nncluster:/data/etag/m05/m05_DaysPriorToMoon.csv')
df5 = ps.read_csv(f'hdfs://nncluster:/data/etag/m05/m05_DaysPriorToNation.csv')
df6 = ps.read_csv(f'hdfs://nncluster:/data/etag/m05/m05_DaysPriorToNewYear.csv')

df2["Date_Start"] = ps.to_datetime(df2["Time"],format = "%Y/%m/%d").dt.strftime('%Y%m%d')
df2['Holiday'] = 'dragon'
df2['Order'] = -1
df2['year'] = df2['Date_Start'].apply(lambda x:str(x)[:4])

df3["Date_Start"] = ps.to_datetime(df3["Time"],format = "%Y/%m/%d").dt.strftime('%Y%m%d')
df3['Holiday'] = 'Lunar New Year'
df3['Order'] = -1
df3['year'] = df3['Date_Start'].apply(lambda x:str(x)[:4])

df4["Date_Start"] = ps.to_datetime(df4["Time"],format = "%Y/%m/%d").dt.strftime('%Y%m%d')
df4['Holiday'] = 'moon'
df4['Order'] = -1
df4['year'] = df4['Date_Start'].apply(lambda x:str(x)[:4])

df5["Date_Start"] = ps.to_datetime(df5["Time"],format = "%Y/%m/%d").dt.strftime('%Y%m%d')
df5['Holiday'] = 'national'
df5['Order'] = -1
df5['year'] = df5['Date_Start'].apply(lambda x:str(x)[:4])

df6["Date_Start"] = ps.to_datetime(df6["Time"],format = "%Y/%m/%d").dt.strftime('%Y%m%d')
df6['Holiday'] = 'New Year'
df6['Order'] = -1
df6['year'] = df6['Date_Start'].apply(lambda x:str(x)[:4])

# concat dataframes
for i in [df2,df3,df4,df5,df6]:
    df = ps.concat([df,i])

# save a parquet(just in case)
df.to_parquet("hdfs://nncluster:/data/etag/m05/m05_holidays_0521_step2.parquet", partition_cols="year")
df = ps.read_parquet(f'hdfs://nncluster:/data/etag/m05/m05_holidays_0521_step2.parquet/')

# remane some stations
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
df = df.groupby(['Hour','Date_Start','Segment', 'Direction','V-Type','Holiday','Order','year','Abnormal']).agg({"Count":"sum","Speed":"mean"}).reset_index()

# save a parquet(just in case)
df.to_parquet("hdfs://nncluster:/data/etag/m05/m05_holidays_0521_step3.parquet", partition_cols="year")
df = ps.read_parquet(f'hdfs://nncluster:/data/etag/m05/m05_holidays_0521_step3.parquet/')

# add year month weekday feature
date = ps.to_datetime(df["Date_Start"],format = "%Y%m%d")

# df["Year"] = date.dt.strftime("%Y").astype(int)
df["Month"] = date.dt.strftime("%m").astype(int)
df["Wday"] = date.dt.strftime("%w").astype(int)

# add segment category feature
df["Seg_cat"] = df["Segment"]
df["Seg_cat"] = df["Seg_cat"].astype("category").cat.codes.astype(int)

# sort data by date & segment & hour
df = df.sort_values(["Date_Start","Seg_cat","Hour"])

# save a parquet(just in case)
df.to_parquet("hdfs://nncluster:/data/etag/m05/m05_holidays_0521_final.parquet", partition_cols="year")
df = ps.read_parquet(f'hdfs://nncluster:/data/etag/m05/m05_holidays_0521_final.parquet/')

# seperate each holiday to parquet
df_lny = df[df['Holiday']=='Lunar New Year']
df_lny.to_parquet("hdfs://nncluster:/data/etag/m05/holidays_lunar_new_year_0521_AFE.parquet", partition_cols="year")


df_d = df[df['Holiday']=='dragon']
df_d.to_parquet("hdfs://nncluster:/data/etag/m05/holidays_dragon_0521_AFE.parquet", partition_cols="year")

df_m = df[df['Holiday']=='moon']
df_m.to_parquet("hdfs://nncluster:/data/etag/m05/holidays_moon_0521_AFE.parquet", partition_cols="year")

df_na = df[df['Holiday']=='national']
df_na.to_parquet("hdfs://nncluster:/data/etag/m05/holidays_national_0521_AFE.parquet", partition_cols="year")

df_ny = df[df['Holiday']=='New Year']
df_ny.to_parquet("hdfs://nncluster:/data/etag/m05/holidays_new_year_0521_AFE.parquet", partition_cols="year")

print("="*50)
print("Feature Engineering Finished !!!")

df_lny = ps.read_parquet("hdfs://nncluster:/data/etag/m05/holidays_lunar_new_year_0521_AFE.parquet")
pdf_lny = df_lny.to_pandas()
pdf_lny.to_csv('holidays_lunar_new_year_0521_AFE.csv')


df_d = ps.read_parquet("hdfs://nncluster:/data/etag/m05/holidays_dragon_0521_AFE.parquet")
pdf_d = df_d.to_pandas()
pdf_d.to_csv('holidays_dragon_0521_AFE.csv')

df_m = ps.read_parquet("hdfs://nncluster:/data/etag/m05/holidays_moon_0521_AFE.parquet")
pdf_m = df_m.to_pandas()
pdf_m.to_csv('holidays_moon_0521_AFE.csv')

df_na = ps.read_parquet("hdfs://nncluster:/data/etag/m05/holidays_national_0521_AFE.parquet")
pdf_na = df_na.to_pandas()
pdf_na.to_csv('holidays_national_0521_AFE.csv')

df_ny = ps.read_parquet("hdfs://nncluster:/data/etag/m05/holidays_new_year_0521_AFE.parquet")
pdf_ny = df_ny.to_pandas()
pdf_ny.to_csv('holidays_new_year_0521_AFE.csv')

