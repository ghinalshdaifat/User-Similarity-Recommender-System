#### als_metrics.py
#!/usr/bin/env python3
import sys
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    split, expr, col, when, collect_list
)
from pyspark.ml.evaluation import RankingEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.recommendation import ALSModel

def mean_reciprocal_rank(pred_and_labels):
    def rr(pair):
        preds, actual = pair
        s = set(actual)
        for i, p in enumerate(preds, start=1):
            if p in s:
                return 1.0 / i
        return 0.0
    return pred_and_labels.map(rr).mean()

def main(preproc_folder, K):
    user = os.environ.get("USER", "unknown")
    base = f"hdfs:///user/{user}/output/{preproc_folder}"

    # grab best params
    local_params = "/tmp/best_params.json"
    os.system(f"hdfs dfs -get -f {base}/best_params.json {local_params}")
    best = json.load(open(local_params, "r"))
    rank, reg = best["rank"], best["regParam"]
    print(f"→ Evaluating model rank={rank}, regParam={reg}, K={K}")

    spark = SparkSession.builder.appName("ALS_Evaluation").getOrCreate()

    # 1) read back your CSV recommendations
    recs_df = (
        spark.read
             .option("header", True)
             .csv(f"{base}/als_recs_K{K}")
             .withColumn("userId", col("userId").cast("int"))
             .withColumn(
                "predicted",
                expr("transform(split(predicted, ','), x -> cast(x as double))")
             )
    )

    # 2) build ground truth from the test set
    test = spark.read.parquet(f"{base}/ratings_test_parquet")
    test_gt = (
        test.groupBy("userId")
            .agg(
               expr("transform(collect_list(movieId), x -> cast(x as double))")
               .alias("groundTruth")
            )
    )

    # 3) join once and persist
    eval_df = recs_df.join(test_gt, on="userId").select("predicted", "groundTruth").persist()

    # 4) ranking metrics
    mapk = RankingEvaluator(
        metricName="meanAveragePrecision",
        labelCol="groundTruth", predictionCol="predicted"
    ).evaluate(eval_df)
    ndcg = RankingEvaluator(
        metricName="ndcgAtK",
        labelCol="groundTruth", predictionCol="predicted"
    ).evaluate(eval_df)

    rdd_pl = eval_df.rdd.map(lambda r: (r.predicted, r.groundTruth))
    rank_metrics = RankingMetrics(rdd_pl)
    precision = rank_metrics.precisionAt(K)
    mrr       = mean_reciprocal_rank(rdd_pl)

    eval_df.unpersist()

    # 5) write out your test_metrics
    out = [(rank, reg, float(mapk), float(ndcg), float(precision), float(mrr))]
    spark.createDataFrame(
        out,
        schema=["rank","regParam","mapk","ndcg","precision","mrr"]
    ).write.mode("overwrite").option("header", True).csv(f"{base}/test_metrics")

    print("Test metrics written to", f"{base}/test_metrics")
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: als_evaluation.py <preproc_folder> [K]")
        sys.exit(1)
    preproc = sys.argv[1]
    K       = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    main(preproc, K)
