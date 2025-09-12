import matplotlib.pyplot as plt

# Metrics from evaluation results
metrics = {
    "neural networks basics": {"ILD@5": 0.825550722058751, "Title-Echo@5": 0.0},
    "data visualization in Python": {"ILD@5": 0.9020305460390521, "Title-Echo@5": 0.0},
    "transformers and attention for NLP": {"ILD@5": 0.8303360634502713, "Title-Echo@5": 0.0},
}

# Prepare data for plotting
queries = list(metrics.keys())
ild_vals = [metrics[q]["ILD@5"] for q in queries]
echo_vals = [metrics[q]["Title-Echo@5"] for q in queries]

x = range(len(queries))
width = 0.35

plt.figure(figsize=(10,6))
plt.bar([i - width/2 for i in x], ild_vals, width=width, label="ILD@5")
plt.bar([i + width/2 for i in x], echo_vals, width=width, label="Title-Echo@5")

plt.xticks(x, queries, rotation=20, ha="right")
plt.ylabel("Score")
plt.title("Evaluation Metrics per Query")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()

# Save figure
plot_path = "evaluation/images/evaluation_metrics_bar.png"
plt.savefig(plot_path)
plot_path

