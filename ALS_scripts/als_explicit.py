#### als_explicit.py
#!/usr/bin/env python3
import sys
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, when, col, concat_ws
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RankingEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import RankingMetrics

def mean_reciprocal_rank(pred_and_labels):
    def rr(pair):
        preds, actual = pair
        s = set(actual)
        for i, p in enumerate(preds, start=1):
            if p in s:
                return 1.0 / i
        return 0.0
    return pred_and_labels.map(rr).mean()


def main(preproc_folder, K, maxIter):
    user = os.environ.get("USER", "unknown")
    base = f"hdfs:///user/{user}/output/{preproc_folder}"

    # Load best params
    local_path = '/tmp/best_params.json'
    os.system(f"hdfs dfs -get -f {base}/best_params.json {local_path}")
    best = json.load(open(local_path))
    rank = best['rank']
    reg  = best['regParam']
    print(f"Using best params rank={rank}, regParam={reg}")

    spark = SparkSession.builder \
        .appName("ALS_Final") \
        .getOrCreate()

    # Read data (train and validation set are read in "full")
    test  = spark.read.parquet(f"{base}/ratings_test_parquet")

    # Train on full train+val
    full = spark.read.parquet(f"{base}/ratings_train_parquet") \
           .union(spark.read.parquet(f"{base}/ratings_validation_parquet"))

    als = ALS(
        userCol="userId", itemCol="movieId", ratingCol="rating",
        rank=rank, regParam=reg, maxIter=maxIter,
        coldStartStrategy="drop", nonnegative=True
    )
    model = als.fit(full)

    # Prepare and save recommendations
    test_gt = test.groupBy("userId").agg(
        expr("transform(collect_list(movieId), x -> cast(x as double))").alias("groundTruth")
    )
    recs = model.recommendForUserSubset(test_gt.select("userId"), K)
    recs = recs.select(
        "userId",
        expr("transform(recommendations, x -> cast(x.movieId as double))").alias("predicted")
    )
    recs_out = recs.select(
        "userId", concat_ws(",", col("predicted")).alias("predicted")
    )
    recs_out.write.mode("overwrite").option("header",True).csv(f"{base}/als_recs_K{K}")
    print(f"Saved recommendations to {base}/als_recs_K{K}")

    spark.stop()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: als_explicit.py <preproc_folder> [K] [maxIter]")
        sys.exit(1)
    preproc = sys.argv[1]
    K       = int(sys.argv[2]) if len(sys.argv)>2 else 100
    maxIter = int(sys.argv[3]) if len(sys.argv)>3 else 10
    main(preproc, K, maxIter)
