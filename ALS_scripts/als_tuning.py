#### als_tuning.py
#!/usr/bin/env python3
import sys
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RankingEvaluator

def main(preproc_folder, ranks, regs, K, maxIter, sampleFrac):
    user = os.environ.get("USER", "unknown")
    base = f"hdfs:///user/{user}/output/{preproc_folder}"

    spark = SparkSession.builder \
        .appName("ALS_Tuning") \
        .getOrCreate()

    # Read pre-split data
    train_full = spark.read.parquet(f"{base}/ratings_train_parquet")
    val_full   = spark.read.parquet(f"{base}/ratings_validation_parquet")

    # Sample for tuning
    train = train_full.sample(False, sampleFrac, seed=42)
    val   = val_full.sample(False, sampleFrac, seed=42)

    # Prepare validation ground truth
    val_gt = val.groupBy("userId").agg(
        expr("transform(collect_list(movieId), x -> cast(x as double))").alias("groundTruth")
    )

    best_map = 0.0
    best_rank = None
    best_reg  = None
    metrics = []

    for rank in ranks:
        for reg in regs:
            print(f"Tuning ALS: rank={rank}, regParam={reg}, maxIter={maxIter}")
            als = ALS(
                userCol="userId", itemCol="movieId", ratingCol="rating",
                rank=rank, regParam=reg, maxIter=maxIter,
                coldStartStrategy="drop", nonnegative=True
            )
            model = als.fit(train)

            recs = model.recommendForUserSubset(val_gt.select("userId"), K)
            recs = recs.select(
                "userId",
                expr("transform(recommendations, x -> cast(x.movieId as double))").alias("predicted")
            )

            df_eval = recs.join(val_gt, on="userId")
            mapk = RankingEvaluator(
                metricName="meanAveragePrecision",
                labelCol="groundTruth",
                predictionCol="predicted"
            ).evaluate(df_eval)

            print(f"Validation MAP@{K}: {mapk:.4f}")
            metrics.append((rank, reg, float(mapk)))

            if mapk > best_map:
                best_map = mapk
                best_rank = rank
                best_reg  = reg

    # Save tuning results
    schema = ["rank","regParam","mapk"]
    spark.createDataFrame(metrics, schema=schema) \
         .write.mode("overwrite").option("header",True) \
         .csv(f"{base}/validation_metrics")
    print(f"Saved validation metrics to {base}/validation_metrics")

    # Save best params as JSON
    best = {"rank": best_rank, "regParam": best_reg, "mapk": best_map}
    out_json = f"{base}/best_params.json"
    # Write locally then copy to HDFS
    with open('/tmp/best_params.json', 'w') as f:
        json.dump(best, f)
    os.system(f"hdfs dfs -put -f /tmp/best_params.json {out_json}")
    print(f"Saved best parameters to {out_json}")

    spark.stop()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: als_tuning.py <preproc_folder> [ranks] [regParams] [K] [maxIter] [sampleFraction]")
        sys.exit(1)
    preproc = sys.argv[1]
    ranks   = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [10,20,50]
    regs    = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.01,0.1,1.0]
    K       = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    maxIter = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    sampleF = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    main(preproc, ranks, regs, K, maxIter, sampleF)
