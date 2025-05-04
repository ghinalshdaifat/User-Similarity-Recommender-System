import matplotlib.pyplot as plt

# MAP@k values from validation set
k_values = list(range(1, 31))
map_scores = [
    0.0023, 0.0020, 0.0126, 0.0099, 0.0093,
    0.0125, 0.0108, 0.0096, 0.0107, 0.0097,
    0.0096, 0.0089, 0.0082, 0.0077, 0.0092,
    0.0095, 0.0090, 0.0086, 0.0090, 0.0086,
    0.0096, 0.0094, 0.0096, 0.0093, 0.0102,
    0.0099, 0.0103, 0.0105, 0.0103, 0.0100
]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(k_values, map_scores, marker='o', linestyle='-', color='steelblue')
plt.xlabel("k")
plt.ylabel("MAP@k")
plt.title("MAP@k Scores on Validation Set")
plt.grid(True)
plt.xticks(k_values)
plt.tight_layout()
plt.savefig("map_at_k_plot.png")
plt.close()

print("Saved: map_at_k_plot.png")