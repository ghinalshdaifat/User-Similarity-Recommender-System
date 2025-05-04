import pandas as pd
import numpy as np
import time
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
from joblib import Parallel, delayed
from scipy.stats import spearmanr

# Configuration 
data_path = 'ml-latest/ratings.csv'
output_csv = '/scratch/gha2009/top_100_movie_twins.csv'
num_perm = 128
lsh_threshold = 0.4

start_time = time.time()

# Load and filter ratings
print("Loading ratings...")
df = pd.read_csv(data_path)

movie_filter = df['movieId'].value_counts()
df = df[df['movieId'].isin(movie_filter[movie_filter >= 100].index)]

user_filter = df['userId'].value_counts()
df = df[df['userId'].isin(user_filter[user_filter >= 200].index)]

user_movie = df.groupby('userId')['movieId'].apply(set)
print(f"Filtered to {len(df)} ratings and {len(user_movie)} users.")

# MinHash signature creation
print("Creating MinHash signatures...")
minhashes = {}
for uid, movies in user_movie.items():
    m = MinHash(num_perm=num_perm)
    for movie in movies:
        m.update(str(movie).encode('utf8'))
    minhashes[uid] = m

# LSH indexing
print("Indexing with MinHashLSH...")
lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
for uid, mh in minhashes.items():
    lsh.insert(str(uid), mh)

# Candidate pair generation
print("Querying similar user pairs...")
pairs = set()
for uid, mh in minhashes.items():
    for neighbor in lsh.query(mh):
        if str(uid) != neighbor:
            pair = tuple(sorted([int(uid), int(neighbor)]))
            pairs.add(pair)
print(f"Generated {len(pairs)} candidate pairs.")

# Compute exact Jaccard distance for candidates
def jaccard_distance(set1, set2):
    return 1 - len(set1 & set2) / len(set1 | set2)

scored_pairs = []
for u1, u2 in pairs:
    dist = jaccard_distance(user_movie[u1], user_movie[u2])
    sim = 1 - dist
    scored_pairs.append(((u1, u2), sim, dist))

top_100_pairs = sorted(scored_pairs, key=lambda x: x[2])[:100]

# Create user rating lookup
user_ratings = defaultdict(dict)
for row in df.itertuples(index=False):
    user_ratings[row.userId][row.movieId] = row.rating

# Compute Spearman correlation
def spearman_corr(u1, u2):
    r1, r2 = user_ratings[u1], user_ratings[u2]
    common = set(r1) & set(r2)
    if len(common) < 2:
        return np.nan
    a = [r1[m] for m in common]
    b = [r2[m] for m in common]
    return spearmanr(a, b).correlation

print("Calculating Spearman correlation (top 100)...")
top_corrs = Parallel(n_jobs=4)(
    delayed(spearman_corr)(u1, u2) for ((u1, u2), i, j) in top_100_pairs
)
top_corrs = [r for r in top_corrs if not np.isnan(r)]
mean_corr_top = np.mean(top_corrs)

# Save results
output_df = pd.DataFrame([
    {'userIdA': u1, 'userIdB': u2, 'JaccardSimilarity': sim, 'JaccardDistance': dist, 'SpearmanCorrelation': corr}
    for ((u1, u2), sim, dist), corr in zip(top_100_pairs, top_corrs)
])
output_df.to_csv(output_csv, index=False)

# Compute baseline correlation for random pairs
print("Calculating baseline Spearman correlation (random pairs)...")
np.random.seed(12299542)
users = list(user_movie.index)
random_pairs = [tuple(np.random.choice(users, 2, replace=False)) for i in range(100)]

rand_corrs = Parallel(n_jobs=4)(
    delayed(spearman_corr)(u1, u2) for u1, u2 in random_pairs
)
rand_corrs = [r for r in rand_corrs if not np.isnan(r)]
mean_corr_rand = np.mean(rand_corrs)

# Final output
print(f"Output written to: {output_csv}")
print(f"Mean Spearman (top 100):   {mean_corr_top:.4f}")
print(f"Mean Spearman (random 100):{mean_corr_rand:.4f}")
print(f"Total runtime: {time.time() - start_time:.2f} seconds")
