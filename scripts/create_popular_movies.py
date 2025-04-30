import sys, os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main(preproc_folder, topK=30, min_count=100):
    spark = SparkSession.builder.appName("CreatePopularMovies").getOrCreate()
    user = os.environ["USER"]
    base = f"hdfs:///user/{user}/output/{preproc_folder}"
    train_path = f"{base}/ratings_train"

    # Load train ratings with numeric rating
    train = spark.read.csv(train_path, header=True, inferSchema=True)

    # Compute average rating and count per movie
    pop = (
        train.groupBy("movieId")
             .agg(
                 F.avg("rating").alias("average_rating"),
                 F.count("rating").alias("count_rating")
             )
             .filter(F.col("count_rating") >= min_count)
             .orderBy(F.col("average_rating").desc())
             .limit(topK)
    )

    # Write out
    out = f"{base}/popular_movies"
    pop.coalesce(1).write.mode("overwrite").option("header", True).csv(out)
    print(f"Wrote popular movies to {out}")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: create_popular_movies.py <preproc_folder> [topK] [min_count]", file=sys.stderr)
        sys.exit(1)
    folder = sys.argv[1]
    topK = int(sys.argv[2]) if len(sys.argv) >= 3 else 30
    min_count = int(sys.argv[3]) if len(sys.argv) == 4 else 100
    main(folder, topK, min_count)
