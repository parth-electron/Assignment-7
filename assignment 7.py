import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


DATA_PATH = "Mall_Customers.csv"

# ---- Task 1: Data Understanding ----
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find '{DATA_PATH}'. Download the dataset from Kaggle "
        f"(vjchoudhary7/customer-segmentation-tutorial-in-python) and place the CSV "
        f"next to this script, or update DATA_PATH above."
    )

df = pd.read_csv(DATA_PATH)

print("First five records:")
print(df.head())
print()

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)
print()

print("Dataset Info:")
df.info()
print()
print("Summary Statistics:")
print(df.describe())

# ---- Task 2: Data Preprocessing ----
print()
print("Missing values per column:")
print(df.isnull().sum())

# Remove unnecessary columns (e.g. CustomerID -- a non-predictive identifier)
cols_to_drop = [c for c in ["CustomerID"] if c in df.columns]
df_model = df.drop(columns=cols_to_drop)
print(f"\nDropped columns: {cols_to_drop}")

# Encode categorical variables (e.g. Gender), if present
cat_cols = df_model.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
print(f"Encoded categorical columns: {cat_cols}")

# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)
print(f"\nFeatures used for clustering: {df_model.columns.tolist()}")

# ---- Task 3: Model Development ----

# 3.1 Elbow Method to determine optimal K
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans_k = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans_k.fit(X_scaled)
    wcss.append(kmeans_k.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), wcss, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.xticks(list(k_range))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("elbow_curve.png", dpi=110)
plt.close()
print("\nElbow curve saved: elbow_curve.png")
print("WCSS values by K:", dict(zip(k_range, [round(w, 2) for w in wcss])))

# ------------------------------------------------------------
# Set OPTIMAL_K by inspecting the elbow curve above.
# For the Mall Customers dataset this is commonly K=5, but
# confirm visually against your own elbow_curve.png before trusting this.
# ------------------------------------------------------------
OPTIMAL_K = 5

# 3.2 Train final KMeans model with chosen K
kmeans = KMeans(n_clusters=OPTIMAL_K, init="k-means++", random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# 3.3 Assign cluster labels to each customer
df["Cluster"] = cluster_labels
print(f"\nCluster sizes:\n{df['Cluster'].value_counts().sort_index()}")

# 3.4 PCA to reduce to 2 principal components (for visualization)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"\nExplained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")
print(f"Total variance captured by 2 components: {pca.explained_variance_ratio_.sum():.4f}")

# ---- Task 4: Visualization and Evaluation ----

# Scatter plot of clusters using two original features (Annual Income vs Spending Score),
# if these columns exist under their standard Kaggle names.
income_col = next((c for c in df.columns if "Income" in c), None)
spend_col = next((c for c in df.columns if "Spending" in c), None)

if income_col and spend_col:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df[income_col], df[spend_col], c=df["Cluster"], cmap="viridis", s=50)
    plt.title(f"Customer Segments ({income_col} vs {spend_col})")
    plt.xlabel(income_col)
    plt.ylabel(spend_col)
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig("cluster_scatter.png", dpi=110)
    plt.close()
    print("\nCluster scatter plot saved: cluster_scatter.png")

# PCA visualization with cluster labels
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="viridis", s=50)
plt.title(f"Customer Clusters Visualized via PCA (K={OPTIMAL_K})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("pca_cluster_visualization.png", dpi=110)
plt.close()
print("PCA cluster visualization saved: pca_cluster_visualization.png")

# Cluster profile summary (mean of original features per cluster) -- useful for observations
profile_cols = [c for c in df.columns if c not in ["CustomerID", "Cluster"] and df[c].dtype != object]
print("\nCluster profile (mean values per cluster):")
print(df.groupby("Cluster")[profile_cols].mean())
