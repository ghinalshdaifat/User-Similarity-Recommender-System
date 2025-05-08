#!/usr/bin/env python3
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.appName("ConvertCSVtoParquet").getOrCreate()
user = os.environ["USER"]
base = f"hdfs:///user/{user}/output/ml-latest_preproc"

for split in ("train", "validation", "test"):
    df = spark.read.csv(f"{base}/ratings_{split}", header=True, inferSchema=True)
    out = f"{base}/ratings_{split}_parquet"
    print(f"Writing {split} Parquet to {out}")
    df.write.mode("overwrite").parquet(out)

spark.stop()
