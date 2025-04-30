import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, collect_list
import pyspark.sql.functions as F

def set_high_rating(df, threshold=3.0):
    return df.withColumn("high_rating", when(col("rating") >= threshold, 1).otherwise(0))

def precision_at_k(recommended, actual, k):
    if k > len(recommended):
        return 0.0
    rel_items = set(actual)
    rec_k = recommended[:k]
    return sum([1 for m in rec_k if m in rel_items]) / k

def compute_map_at_k(df, popular_list, k):
    high = set_high_rating(df)
    user_actual = (
        high.filter(col("high_rating") == 1)
            .groupBy("userId")
            .agg(collect_list("movieId").alias("movieId_list"))
            .rdd.map(lambda row: (row.userId, row.movieId_list))
    )
    per_user = user_actual.map(lambda ua: precision_at_k(popular_list, ua[1], k))
    vals = per_user.collect()
    return sum(vals) / len(vals) if vals else 0.0

def main(preproc_folder, popular_folder):
    spark = SparkSession.builder.appName("EvalPopularityModel").getOrCreate()
    user = os.environ["USER"]
    base = f"hdfs:///user/{user}/output/{preproc_folder}"

    # Load popular movies: movieId, average_rating, count_rating
    pop = (
        spark.read
             .csv(f"{base}/{popular_folder}", header=True, inferSchema=True)
             .orderBy(F.col("average_rating").desc())
    )
    popular_list = [row.movieId for row in pop.collect()]

    # Load validation and test ratings
    val = spark.read.csv(f"{base}/ratings_validation", header=True, inferSchema=True)
    test = spark.read.csv(f"{base}/ratings_test", header=True, inferSchema=True)

    # Compute MAP@k for k=1..30 on validation
    best_k = 1
    best_map = 0.0
    for k in range(1, 31):
        m = compute_map_at_k(val, popular_list, k)
        print(f"MAP@{k}: {m:.4f}")
        if m > best_map:
            best_map = m
            best_k = k
    print(f"Best k: {best_k}")

    # Evaluate on test
    m_test = compute_map_at_k(test, popular_list, best_k)
    print(f"MAP@{best_k} on test dataset: {m_test:.4f}")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: evaluate_popularity_model.py <preproc_folder> <popular_folder>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
