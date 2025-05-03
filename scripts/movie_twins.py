import pandas as pd
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, udf, col
from pyspark.ml.feature import CountVectorizer, MinHashLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import ArrayType, StringType

# Function to load ratings data from HDFS 
def load_data(spark, user_id, folder_name):
    path = f'hdfs:/user/{user_id}/target/{folder_name}/ratings.csv'
    # Read the ratings file, drop rating and timestamp columns since they are not needed for movie sets
    ratings_df = spark.read.csv(path, header=True, inferSchema=True)
    ratings_df = ratings_df.drop('rating', 'timestamp')
    return ratings_df

# Function to group movies rated by each user 
def preprocess_movies(ratings_df):
    # Group by userId, collect all movieIds as a set
    movies_by_user = ratings_df.groupBy("userId").agg(collect_set("movieId").alias("movies"))

    # Convert movieIds from integers to strings (needed for CountVectorizer)
    to_string_array = udf(lambda movie_list: [str(mid) for mid in movie_list], ArrayType(StringType()))
    movies_by_user = movies_by_user.withColumn("movies", to_string_array("movies"))
    
    return movies_by_user

#  Function to vectorize movie sets into binary feature vectors 
def generate_features(movies_by_user):
    # Use binary CountVectorizer to create feature vectors (presence/absence of each movie)
    vectorizer = CountVectorizer(inputCol="movies", outputCol="features", binary=True)
    model = vectorizer.fit(movies_by_user)
    vectorized_data = model.transform(movies_by_user)
    
    return vectorized_data

# Function to find similar users using MinHash LSH 
def find_similar_users(vectorized_data):
    # Initialize MinHashLSH
    minhash = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    mh_model = minhash.fit(vectorized_data)

    # Approximate similarity join within the dataset
    similar_pairs = mh_model.approxSimilarityJoin(vectorized_data, vectorized_data, threshold=0.5, distCol="JaccardDistance")
    
    # Filter to avoid self-pairs and duplicates (only userIdA > userIdB)
    similar_pairs = similar_pairs.filter(col("datasetA.userId") > col("datasetB.userId"))
    
    # Take the top 100 most similar user pairs (smallest Jaccard distance)
    top_100_pairs = similar_pairs.orderBy("JaccardDistance", ascending=True).limit(100)
    
    return top_100_pairs

#  Main execution function
def main():
    # Initialize Spark session with memory settings
    spark = SparkSession.builder \
    .appName("MovieTwinsLSH") \
    .config("spark.executor.memory", "24g") \
    .config("spark.executor.memoryOverhead", "8g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

    # Get current user's ID and input folder name from arguments
    user_id = os.environ['USER']
    input_folder = sys.argv[1]

    # Load and process data 
    ratings_df = load_data(spark, user_id, input_folder)
    movies_by_user = preprocess_movies(ratings_df)
    vectorized_data = generate_features(movies_by_user)
    top_100_pairs = find_similar_users(vectorized_data)

    # Show sample output in console 
    top_100_pairs.show()

    # Save top 100 movie twins to CSV
    output_path = f"top_100_{input_folder}.csv"
    top_100_pairs_pd = top_100_pairs.select(
        col("datasetA.userId").alias("userIdA"),
        col("datasetB.userId").alias("userIdB"),
        col("JaccardDistance")
    ).toPandas()
    top_100_pairs_pd.to_csv(output_path, index=False)

    # Save to HDFS
    top_100_pairs.select(
        col("datasetA.userId").alias("userIdA"),
        col("datasetB.userId").alias("userIdB"),
        col("JaccardDistance")
    ).write.mode("overwrite").csv(f"hdfs:///user/gha2009_nyu_edu/output/top_100_{input_folder}", header=True)
    spark.stop()

if __name__ == "__main__":
    main()

