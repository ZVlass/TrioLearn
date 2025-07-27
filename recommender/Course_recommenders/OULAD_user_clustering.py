from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



oulad_df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')

# Select proportion features for clustering
oulad_features = oulad_df[['course_prop', 'reading_prop', 'video_prop']].fillna(0)

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(oulad_features)

# Apply KMeans clustering (e.g., 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
oulad_df['user_cluster'] = kmeans.fit_predict(scaled_features)

# Show cluster counts
cluster_counts = oulad_df['user_cluster'].value_counts().sort_index()

# Show sample users per cluster
cluster_samples = oulad_df.groupby('user_cluster').head(2)
#print(cluster_samples, cluster_counts)

# save to csv fiel
oulad_df[['id_student', 'course_prop', 'reading_prop', 'video_prop', 'user_cluster']].to_csv(
    "./data/interim/oulad_clustered.csv", index=False
)


# Reduce dimensions with PCA for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled_features)
oulad_df['PC1'] = reduced[:, 0]
oulad_df['PC2'] = reduced[:, 1]

# Plot clusters
plt.figure(figsize=(8, 6))
for cluster in sorted(oulad_df['user_cluster'].unique()):
    cluster_data = oulad_df[oulad_df['user_cluster'] == cluster]
    plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}', alpha=0.6)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('User Clusters from OULAD Media Profiles (PCA 2D Projection)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
