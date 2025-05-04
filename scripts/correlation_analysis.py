import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from collections import defaultdict

# Load your top-100 results
df = pd.read_csv("top_100_movie_twins.csv")
top_corrs = df["SpearmanCorrelation"].dropna().tolist()

# Load full ratings and build user -> {movieId: rating} map
ratings = pd.read_csv("/teamspace/studios/this_studio/capstone-bdcs-10/scripts/ratings.csv")
user_ratings = defaultdict(dict)
for row in ratings.itertuples(index=False):
    user_ratings[row.userId][row.movieId] = row.rating

all_users = list(user_ratings.keys())

# Spearman correlation function
def spearman_corr(u1, u2):
    r1, r2 = user_ratings[u1], user_ratings[u2]
    overlap = set(r1) & set(r2)
    if len(overlap) < 2:
        return np.nan
    a = [r1[m] for m in overlap]
    b = [r2[m] for m in overlap]
    return spearmanr(a, b).correlation

# Generate random pairs
np.random.seed(12299542)
rand_pairs = [tuple(np.random.choice(all_users, 2, replace=False)) for i in range(100)]

# Compute correlations for random pairs
rand_corrs = Parallel(n_jobs=4)(
    delayed(spearman_corr)(u1, u2) for u1, u2 in rand_pairs
)
rand_corrs = [r for r in rand_corrs if not np.isnan(r)]

# Run t-test
t_stat, p_val = ttest_ind(top_corrs, rand_corrs, equal_var=False)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value:     {p_val:.4e}")

# Plot KDE distributions
plt.figure(figsize=(8, 5))
sns.kdeplot(top_corrs, label='Top 100', fill=True)
sns.kdeplot(rand_corrs, label='Random 100', fill=True)
plt.xlabel('Spearman Correlation')
plt.title('Distribution of Spearman Correlations')
plt.legend()
plt.savefig("spearman_kdeplot.png")
plt.close()
