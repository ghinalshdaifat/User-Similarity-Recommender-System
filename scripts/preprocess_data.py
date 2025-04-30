
import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, collect_list, expr,
    row_number, when
)
from pyspark.sql.window import Window
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, DoubleType, StringType
)

def main(input_folder):
    # 1) Spark setup & paths
    spark = SparkSession.builder \
        .appName("PreprocessAllData") \
        .getOrCreate()

    user     = os.environ["USER"]
    hdfs_in  = f"hdfs:///user/{user}/target/{input_folder}"
    out_base = f"hdfs:///user/{user}/output/{input_folder}_preproc"

    # 2) Explicit schemas
    ratings_schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", DoubleType(), True),
        StructField("timestamp_rating", IntegerType(), True),
    ])
    tags_schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("tag", StringType(), True),
        StructField("timestamp_tag", IntegerType(), True),
    ])
    movies_schema = StructType([
        StructField("movieId", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("genres", StringType(), True),
    ])
    genome_scores_schema = StructType([
        StructField("movieId", IntegerType(), True),
        StructField("tagId", IntegerType(), True),
        StructField("relevance", DoubleType(), True),
    ])
    genome_tags_schema = StructType([
        StructField("tagId", IntegerType(), True),
        StructField("tag", StringType(), True),
    ])

    # 3) Load & clean raw tables
    ratings      = spark.read.csv(f"{hdfs_in}/ratings.csv",
                                  header=True, schema=ratings_schema) \
                             .dropna().dropDuplicates()
    tags         = spark.read.csv(f"{hdfs_in}/tags.csv",
                                  header=True, schema=tags_schema) \
                             .dropna().dropDuplicates()
    movies       = spark.read.csv(f"{hdfs_in}/movies.csv",
                                  header=True, schema=movies_schema) \
                             .dropna().dropDuplicates()
    genome_scores= spark.read.csv(f"{hdfs_in}/genome-scores.csv",
                                  header=True, schema=genome_scores_schema) \
                             .dropna().dropDuplicates()
    genome_tags  = spark.read.csv(f"{hdfs_in}/genome-tags.csv",
                                  header=True, schema=genome_tags_schema) \
                             .dropna().dropDuplicates()

    # 4) Build genome‐score aggregates per movie
    gs_with_tag = genome_scores.join(genome_tags, "tagId") \
        .select("movieId","tag","relevance")
    gs_agg = gs_with_tag.groupBy("movieId").agg(
        collect_list("tag").alias("genome_tags"),
        expr("percentile_approx(relevance, 0.5)")
            .alias("median_relevance")
    )

    # 5) Enrich movies with genome + genre features
    movies_enriched = (
        movies
        .join(gs_agg,   on="movieId", how="left")       # keep all movies
        .withColumn("genres_array", split(col("genres"), "\\|"))
    )
    cv_model = CountVectorizer(
        inputCol="genres_array",
        outputCol="genreFeatures"
    ).fit(movies_enriched)
    movies_enriched = cv_model.transform(movies_enriched)

    # write it out once
    movies_enriched.write.mode("overwrite") \
                   .parquet(f"{out_base}/movies_enriched")

    # 6) Chronological train/val/test split of ratings per user
    w = Window.partitionBy("userId").orderBy("timestamp_rating")
    rated = ratings.withColumn("rank", row_number().over(w))
    counts= rated.groupBy("userId") \
                 .count().withColumnRenamed("count","total")
    rated = rated.join(counts, on="userId")

    split_col = when(col("rank") <= col("total")*0.6, "train") \
              .when(col("rank") <= col("total")*0.8, "validation") \
              .otherwise("test")
    rated = rated.withColumn("split", split_col)

    # write each out
    for split_name in ("train","validation","test"):
        rated.filter(col("split")==split_name) \
             .select("userId","movieId","rating","timestamp_rating") \
             .write \
             .mode("overwrite") \
             .option("header",True) \
             .csv(f"{out_base}/ratings_{split_name}")

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: preprocess_data.py <ml-folder>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
