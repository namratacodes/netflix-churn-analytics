import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def scale_features(df, feature_cols):
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return X_scaled, scaler


def evaluate_k_range(X_scaled, k_range=range(2, 9), random_state=42):
    
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        results.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X_scaled, labels),
        })
    return pd.DataFrame(results)


def fit_kmeans(df, feature_cols, k, random_state=42):
    
    X_scaled, scaler = scale_features(df, feature_cols)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)

    df = df.copy()
    df["cluster"] = kmeans.fit_predict(X_scaled)

    return df, kmeans, scaler


def profile_clusters(df, feature_cols, cluster_col="cluster"):
    
    profile = df.groupby(cluster_col)[feature_cols].mean().round(2)
    profile["size"] = df[cluster_col].value_counts().sort_index()
    return profile


def validate_against_ground_truth(df, cluster_col="cluster", label_col="persona"):
    
    return pd.crosstab(df[cluster_col], df[label_col])


def assign_segment_names(df, cluster_name_map, cluster_col="cluster", new_col="segment"):
   
    df = df.copy()
    df[new_col] = df[cluster_col].map(cluster_name_map)
    return df