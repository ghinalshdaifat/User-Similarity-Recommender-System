# Movie Recommender System & User Segmentation at Scale

A large-scale recommender system built on the [MovieLens](https://grouplens.org/datasets/movielens/latest/) dataset (33 million ratings, 86,000 movies, 330,000 users). This project combines locality-sensitive hashing for user segmentation with a collaborative filtering model using Alternating Least Squares (ALS), all processed on NYU's High Performance Computing (HPC) cluster via Apache Spark.

**Authors:** Ghina Al Shdaifat, Hamza Alshamy

---

## Overview

This project addresses two core problems in large-scale recommendation:

1. **Customer Segmentation** — Identifying the top 100 most similar user pairs ("movie twins") based on shared movie-watching behavior using MinHash-based Locality Sensitive Hashing (LSH), and validating those matches through rating correlation analysis.

2. **Movie Recommendation** — Building and evaluating a personalized recommendation system using Spark's ALS collaborative filtering model, benchmarked against a non-personalized popularity baseline.

---

## Repository Structure

```
├── ALS_scripts/
│   ├── Parquet_files/
│   │   ├── ratings_train_parquet/
│   │   ├── ratings_validation_parquet/
│   │   └── ratings_test_parquet/
│   ├── als_explicit.py          # ALS model using best hyperparameters
│   ├── als_metrics.py           # Evaluation metrics for the final ALS model
│   ├── als_tuning.py            # Hyperparameter grid search (rank, regularization)
│   ├── convert_to_parquet.py    # Converts CSV splits to Parquet for faster I/O
│   ├── test_metrics.csv         # Final test set evaluation results
│   └── validation_metrics.csv  # Validation set evaluation results
│
├── scripts/
│   ├── customer_segmentation_final.py      # MinHashLSH pipeline for user similarity
│   ├── run_customer_segmentation_final.sh  # SLURM batch job submission script
│   ├── correlation_analysis.py             # Spearman correlation + t-test validation
│   ├── preprocess_data.py                  # Time-aware train/val/test splitting
│   ├── create_popular_movies.py            # Popularity baseline model
│   ├── popularity_baseline_analysis.py     # MAP@k evaluation for baseline
│   ├── evaluate_popularity_model.py        # Popularity model evaluation script
│   ├── top_100_movie_twins.csv             # Output: top 100 most similar user pairs
│   ├── map_at_k_plot.png                   # MAP@k curve visualization
│   └── spearman_kdeplot.png                # Spearman correlation distribution plot
│
└── Capstone_Report.pdf                     # Full project report
```

---

## Part 1: Customer Segmentation

### Approach

We used the full `ml-latest/ratings.csv` dataset and applied the following preprocessing to improve signal quality and computational efficiency:

- Retained only users who rated at least **200 movies**
- Retained only movies with at least **100 ratings**

After filtering: **20,525,899 ratings** from **42,524 users**.

Each user was represented as a set of `movieId`s they had rated — capturing their watching "style" independently of actual rating values.

### MinHash + LSH

We generated MinHash signatures (128 permutations) for each user using the `datasketch` library and indexed them with `MinHashLSH` at a similarity threshold of 0.4. This threshold was selected empirically: it produced 10,390,581 candidate pairs while keeping runtime tractable (~1329 seconds on the full dataset).

For each candidate pair, we computed the exact Jaccard similarity between their movie sets and selected the **top 100 pairs** by highest similarity.

### Validation via Spearman Correlation

To confirm that users with overlapping watch histories also rate movies similarly, we computed Spearman rank correlations for each of the top 100 pairs and compared them to 100 randomly selected user pairs:

| Group | Mean Spearman Correlation |
|---|---|
| Top 100 similar pairs | **0.8737** |
| Random 100 pairs | **0.1652** |

An independent two-sample t-test confirmed the difference is highly statistically significant:

- **T-statistic:** 6.2166
- **P-value:** 4.85 × 10⁻⁷

This validates that MinHash-based segmentation identifies users who not only share similar watching behavior but also agree on how they rate movies.

> Spearman correlation was chosen over Pearson because movie ratings are ordinal data, and Spearman is more robust to outliers and does not assume linear relationships — making it better suited to capture monotonic agreement in human rating behavior.

---

## Part 2: Movie Recommendation

### Data Preprocessing & Splitting

All splits were performed with a **time-aware, per-user chronological strategy** to prevent data leakage:

- First 60% of each user's ratings (by timestamp) → **training set**
- Next 20% → **validation set**
- Final 20% → **test set**

This ensures the model is always evaluated on ratings that occur after the training data, reflecting a realistic deployment scenario. The splits were converted to **Parquet format** for efficient I/O during training and evaluation (stored in `ALS_scripts/Parquet_files/`).

> Note: The raw CSV splits exceed GitHub's 100 MB file limit and are stored on HDFS only.

### Popularity Baseline

A non-personalized popularity model was implemented as a lower-bound baseline. It recommends the top-K globally highest-rated movies (filtered to those with ≥100 ratings) to all users.

Evaluating MAP@k for k = 1–100 on the validation set showed peak performance at k = 3, with MAP approaching zero by k = 100 — consistent with the expected behavior of a non-personalized model on a large sparse dataset.

### ALS Latent Factor Model

We implemented collaborative filtering using Spark's `pyspark.ml.recommendation.ALS` module with explicit ratings. Hyperparameters were tuned via grid search over rank and regularization (λ) on a 40% subsample of the training data, evaluated by MAP@100 on the validation set.

**Tuning summary:**

| Iteration | Best Rank | Best λ | MAP@100 |
|---|---|---|---|
| 1st | 10 | 0.01 | 0.0000 |
| 2nd | 60 | 0.05 | 0.0019 |
| 3rd | 70 | 0.05 | 0.0020 |
| 4th (final) | **90** | **0.05** | **0.0026** |

Key findings from tuning: λ = 0.05 consistently outperformed stronger regularization at every rank, and MAP@100 increased monotonically with rank when λ was fixed.

### Final Model Evaluation (rank=90, λ=0.05)

| Metric | Validation | Test |
|---|---|---|
| MAP@100 | 0.0026 | 0.0030 |
| NDCG@100 | — | 0.0040 |
| Precision@100 | — | 0.0011 |
| MRR | — | 0.0052 |

While absolute MAP values are low — a known challenge on large, sparse datasets with explicit ratings — ALS substantially outperforms the non-personalized baseline and demonstrates the ability to surface long-tail, personalized recommendations.

---

## Infrastructure

All experiments were run on **NYU's HPC cluster** using SLURM batch jobs and Apache Spark for distributed computation. The customer segmentation job was configured with 4 CPU cores, 64 GB memory, and a 4-hour runtime limit.

---

## Dependencies

- Apache Spark (PySpark) 3.x
- `datasketch` (MinHash + LSH)
- `scipy` (Spearman correlation, t-test)
- `pandas`, `matplotlib`, `seaborn`

---

## Full Report

For complete implementation details, hyperparameter tables, and analysis, see [`WrittenReport.pdf`](./Capstone_Report.pdf).
