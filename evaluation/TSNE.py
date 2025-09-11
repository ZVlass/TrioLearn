import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
import seaborn as sns
import numpy as np

# Simulate SBERT embeddings and LDA topic labels
np.random.seed(42)
X, labels = make_blobs(n_samples=200, centers=5, n_features=384, cluster_std=5.0)

# Reduce to 2D with t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Convert to DataFrame
df_tsne = pd.DataFrame({
    "x": X_tsne[:, 0],
    "y": X_tsne[:, 1],
    "Topic": labels
})

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_tsne, x="x", y="y", hue="Topic", palette="tab10", s=60)
plt.title("t-SNE Visualization of Course Embeddings by LDA Topic")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title="Topic")
plt.grid(True)
plt.tight_layout()
plt.show()
