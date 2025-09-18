from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_kmeans(
    df: pd.DataFrame,
    feature_columns: List[str],
    k: int = 3,
    max_iter: int = 300,
    random_state: int = 42,
) -> Tuple[pd.Series, float]:
    from sklearn.cluster import KMeans

    features = df[feature_columns] if feature_columns else df.select_dtypes(include=["number"])  # fallback
    model = KMeans(n_clusters=k, max_iter=max_iter, n_init=10, random_state=random_state)
    labels = model.fit_predict(features)
    inertia = model.inertia_
    return pd.Series(labels, index=df.index, name="cluster"), float(inertia)


def plot_clusters(df: pd.DataFrame, feature_columns: List[str], labels: pd.Series, k: int):
    """Vẽ biểu đồ phân cụm 2D hoặc 3D tùy số lượng đặc trưng."""
    if len(feature_columns) < 2:
        return None
    
    plt.figure(figsize=(10, 6))
    
    if len(feature_columns) == 2:
        # Biểu đồ 2D
        scatter = plt.scatter(df[feature_columns[0]], df[feature_columns[1]], 
                             c=labels, cmap='viridis', alpha=0.7)
        plt.xlabel(feature_columns[0])
        plt.ylabel(feature_columns[1])
        plt.title(f'K-means Clustering (k={k})')
        plt.colorbar(scatter, label='Cluster')
        
    elif len(feature_columns) >= 3:
        # Biểu đồ 3D
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(df[feature_columns[0]], df[feature_columns[1]], df[feature_columns[2]], 
                           c=labels, cmap='viridis', alpha=0.7)
        ax.set_xlabel(feature_columns[0])
        ax.set_ylabel(feature_columns[1])
        ax.set_zlabel(feature_columns[2])
        ax.set_title(f'K-means Clustering 3D (k={k})')
        plt.colorbar(scatter, label='Cluster')
    
    plt.tight_layout()
    return plt.gcf()


def plot_elbow_method(df: pd.DataFrame, feature_columns: List[str], max_k: int = 10):
    """Vẽ biểu đồ elbow method để chọn k tối ưu."""
    from sklearn.cluster import KMeans
    
    features = df[feature_columns] if feature_columns else df.select_dtypes(include=["number"])
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(features)
        inertias.append(model.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Số cụm (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method - Chọn số cụm tối ưu')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_cluster_stats(df: pd.DataFrame, feature_columns: List[str], labels: pd.Series):
    """Vẽ biểu đồ thống kê theo cụm."""
    if not feature_columns:
        return None
        
    # Tính thống kê theo cụm
    cluster_stats = df[feature_columns + ['cluster']].groupby('cluster').mean()
    
    plt.figure(figsize=(12, 6))
    cluster_stats.plot(kind='bar', ax=plt.gca())
    plt.title('Thống kê trung bình theo cụm')
    plt.xlabel('Cụm')
    plt.ylabel('Giá trị trung bình')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return plt.gcf()

