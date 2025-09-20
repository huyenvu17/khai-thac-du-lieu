from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_kmeans(
    df: pd.DataFrame,
    k: int = 4,
    max_iter: int = 300,
    random_state: int = 42,
) -> Tuple[pd.Series, float, dict]:
    """
    Phân nhóm khách hàng theo Purchase Amount và Review Rating.
    
    Args:
        df: DataFrame chứa dữ liệu khách hàng
        k: Số cụm
        max_iter: Số lần lặp tối đa
        random_state: Random seed
        
    Returns:
        Tuple[labels, inertia, cluster_info]: Nhãn cụm, inertia, và thông tin cụm
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Chuẩn bị dữ liệu
    data = df.dropna(subset=['Purchase Amount (USD)', 'Review Rating']).copy()
    
    # Chọn 2 features chính: Purchase Amount và Review Rating
    feature_columns = ['Purchase Amount (USD)', 'Review Rating']
    features = data[feature_columns]
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Chạy K-means
    model = KMeans(n_clusters=k, max_iter=max_iter, n_init=10, random_state=random_state)
    labels = model.fit_predict(features_scaled)
    inertia = model.inertia_
    
    # Tạo thông tin cụm
    data['cluster'] = labels
    cluster_info = _generate_cluster_insights(data, feature_columns)
    
    return pd.Series(labels, index=data.index, name="cluster"), float(inertia), cluster_info


def _generate_cluster_insights(df: pd.DataFrame, feature_columns: List[str]) -> dict:
    """Tạo insights về các cụm khách hàng."""
    insights = {}
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Tính thống kê cơ bản
        avg_amount = cluster_data['Purchase Amount (USD)'].mean()
        avg_rating = cluster_data['Review Rating'].mean()
        count = len(cluster_data)
        
        # Phân loại cụm
        if avg_amount >= 80 and avg_rating >= 4.0:
            cluster_type = "VIP - Chi tiêu cao, hài lòng"
        elif avg_amount >= 80 and avg_rating < 4.0:
            cluster_type = "Chi tiêu cao, chưa hài lòng"
        elif avg_amount < 80 and avg_rating >= 4.0:
            cluster_type = "Trung thành - Chi tiêu vừa, hài lòng"
        else:
            cluster_type = "Cần quan tâm - Chi tiêu thấp, chưa hài lòng"
        
        insights[cluster_id] = {
            'count': count,
            'avg_amount': round(avg_amount, 2),
            'avg_rating': round(avg_rating, 2),
            'type': cluster_type,
            'percentage': round(count / len(df) * 100, 1)
        }
    
    return insights


def plot_clusters(df: pd.DataFrame, labels: pd.Series, k: int):
    """Vẽ biểu đồ phân cụm khách hàng theo Purchase Amount và Review Rating."""
    plt.figure(figsize=(12, 8))
    
    # Biểu đồ 2D: Purchase Amount vs Review Rating
    scatter = plt.scatter(df['Purchase Amount (USD)'], df['Review Rating'], 
                         c=labels, cmap='viridis', alpha=0.7, s=60)
    
    plt.xlabel('Purchase Amount (USD)', fontsize=12)
    plt.ylabel('Review Rating', fontsize=12)
    plt.title(f'Phân Nhóm Khách Hàng - K-means (k={k})\nTheo Mức Chi Tiêu và Độ Hài Lòng', fontsize=14, fontweight='bold')
    
    # Thêm colorbar
    cbar = plt.colorbar(scatter, label='Nhóm Khách Hàng')
    cbar.set_ticks(range(k))
    cbar.set_ticklabels([f'Nhóm {i}' for i in range(k)])
    
    # Thêm grid và styling
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_elbow_method(df: pd.DataFrame, max_k: int = 10, highlight_k: int = None):
    """Vẽ biểu đồ elbow method để chọn k tối ưu cho phân nhóm khách hàng."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Chuẩn bị dữ liệu
    data = df.dropna(subset=['Purchase Amount (USD)', 'Review Rating']).copy()
    features = data[['Purchase Amount (USD)', 'Review Rating']]
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    inertias = []
    k_range = range(1, max_k + 1)
    
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(features_scaled)
        inertias.append(model.inertia_)
    
    plt.figure(figsize=(12, 6))
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Số Nhóm Khách Hàng (k)', fontsize=12)
    plt.ylabel('Inertia (Tổng Bình Phương Khoảng Cách)', fontsize=12)
    plt.title('Elbow Method - Chọn Số Nhóm Khách Hàng Tối Ưu', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Đánh dấu điểm k được chọn
    if highlight_k and 1 <= highlight_k <= max_k:
        plt.axvline(x=highlight_k, color='red', linestyle='-', linewidth=3, alpha=0.8, 
                   label=f'K đã chọn = {highlight_k}')
        plt.legend()
    
    # Đánh dấu điểm gợi ý (k=3 và k=4) nếu không có highlight_k
    elif len(inertias) >= 4:
        plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Gợi ý k=3')
        plt.axvline(x=4, color='orange', linestyle='--', alpha=0.7, label='Gợi ý k=4')
        plt.legend()
    
    plt.tight_layout()
    return plt.gcf()


def plot_cluster_stats(df: pd.DataFrame, labels: pd.Series, cluster_info: dict):
    """Vẽ biểu đồ thống kê và insights về các nhóm khách hàng."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Biểu đồ 1: Thống kê trung bình theo nhóm
    df_with_cluster = df.copy()
    df_with_cluster['cluster'] = labels
    
    cluster_stats = df_with_cluster.groupby('cluster')[['Purchase Amount (USD)', 'Review Rating']].mean()
    
    cluster_stats.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Thống Kê Trung Bình Theo Nhóm Khách Hàng', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Nhóm Khách Hàng')
    ax1.set_ylabel('Giá Trị Trung Bình')
    ax1.legend(['Chi Tiêu (USD)', 'Đánh Giá'])
    ax1.grid(True, alpha=0.3)
    
    # Biểu đồ 2: Phân bố số lượng khách hàng theo nhóm
    cluster_counts = [cluster_info[i]['count'] for i in sorted(cluster_info.keys())]
    cluster_labels = [f'Nhóm {i}\n({cluster_info[i]["type"]})' for i in sorted(cluster_info.keys())]
    
    bars = ax2.bar(range(len(cluster_counts)), cluster_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_title('Phân Bố Số Lượng Khách Hàng Theo Nhóm', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Nhóm Khách Hàng')
    ax2.set_ylabel('Số Lượng Khách Hàng')
    ax2.set_xticks(range(len(cluster_counts)))
    ax2.set_xticklabels(cluster_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Thêm số liệu trên cột
    for i, (bar, count) in enumerate(zip(bars, cluster_counts)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{count}\n({cluster_info[i]["percentage"]}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

