import os
import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, rand

def main(input_folder):
    spark = SparkSession.builder \
        .appName("CapstoneCorrelation") \
        .getOrCreate()

    # Base HDFS path 
    user = os.environ["USER"]
    hdfs_base = f"hdfs:///user/{user}"

    # 1) Load top-100 “movie twins” from Q1
    top100 = (
        spark.read
             .option("header", "true")
             .csv(f"{hdfs_base}/output/top_100_{input_folder}")
             .select(
                 col("userIdA").cast("int"),
                 col("userIdB").cast("int")
             )
    )

    # 2) Load the full ratings dataset
    ratings = (
        spark.read
             .option("header", "true")
             .csv(f"{hdfs_base}/target/{input_folder}/ratings.csv")
             .select(
                 col("userId").cast("int"),
                 col("movieId").cast("int"),
                 col("rating").cast("double")
             )
    )

    # 3) Prepare two side-tables to avoid ambiguous column names
    ratings_a = ratings.select(
        col("userId").alias("userIdA"),
        col("movieId"),
        col("rating").alias("ra")
    )
    ratings_b = ratings.select(
        col("userId").alias("userIdB"),
        col("movieId"),
        col("rating").alias("rb")
    )

    # 4) Join top100 : ratings_a on userIdA : ratings_b on (userIdB, movieId)
    joined_top = (
        top100
        .join(ratings_a, on="userIdA", how="inner")
        .join(ratings_b, on=["userIdB", "movieId"], how="inner")
    )

    # 5) Compute Pearson for each pair and then average
    corrs_top = joined_top.groupBy("userIdA", "userIdB") \
                          .agg(F.corr("ra", "rb").alias("pearson"))
    avg_top = corrs_top.agg(F.avg("pearson")).first()[0]
    print(f"Average Pearson correlation (top 100): {avg_top:.4f}")

    # 6) Build 100 random pairs by sampling 200 distinct users
    users = ratings.select("userId").distinct()
    sample_ids = users.orderBy(rand()).limit(200).rdd.map(lambda r: r[0]).collect()
    rand_list = [(sample_ids[i], sample_ids[i+1]) for i in range(0, 200, 2)]
    rand_pairs = spark.createDataFrame(rand_list, schema=["userIdA", "userIdB"])

    joined_rand = (
        rand_pairs
        .join(ratings_a, on="userIdA", how="inner")
        .join(ratings_b, on=["userIdB", "movieId"], how="inner")
    )

    corrs_rand = joined_rand.groupBy("userIdA", "userIdB") \
                            .agg(F.corr("ra", "rb").alias("pearson"))
    avg_rand = corrs_rand.agg(F.avg("pearson")).first()[0]
    print(f"Average Pearson correlation (100 random): {avg_rand:.4f}")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: correlation.py <ml-folder>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
