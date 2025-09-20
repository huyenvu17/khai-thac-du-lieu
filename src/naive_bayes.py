from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def run_nb(
    df: pd.DataFrame,
    target: str = None,
    feature_columns: List[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict, pd.Series]:
    """
    Naive Bayes: Dự đoán khả năng mua lại của khách hàng.
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import confusion_matrix
    from .utils import train_test_split_df
    from .utils import compute_classification_metrics

    # Bài toán: Dự đoán khả năng mua lại của khách hàng
    df_processed = df.copy()
    
    # Kiểm tra xem có cột Will_Return chưa
    if 'Will_Return' in df_processed.columns:
        # Sử dụng cột có sẵn
        target = 'Will_Return'
    else:
        # Tạo target dựa trên logic: Rating cao + chi tiêu hợp lý = sẽ mua lại
        avg_amount = df_processed['Purchase Amount (USD)'].mean()
        df_processed['Will_Return'] = (
            (df_processed['Review Rating'] >= 4.0) & 
            (df_processed['Purchase Amount (USD)'] >= avg_amount * 0.8)  # Giảm ngưỡng để có nhiều dữ liệu
        ).astype(int)
        target = 'Will_Return'
    
    # Features chính
    feature_columns = ['Review Rating', 'Purchase Amount (USD)', 'Item Purchased', 'Payment Method']
    
    # Chuẩn bị dữ liệu
    used_df = df_processed[feature_columns + [target]].dropna()

    # Xử lý categorical features
    processed_df = used_df.copy()
    label_encoders = {}
    
    for col in feature_columns:
        if used_df[col].dtype == 'object':  # Categorical column
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(used_df[col].astype(str))
            label_encoders[col] = le

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split_df(
        processed_df, target=target, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = GaussianNB(var_smoothing=1e-9)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Tính metrics
    cm = confusion_matrix(y_test, y_pred)
    metrics = compute_classification_metrics(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["features"] = feature_columns
    metrics["avg_amount"] = df_processed['Purchase Amount (USD)'].mean()
    
    # Tạo bảng xác suất theo format ví dụ
    probability_table = _create_probability_table(used_df, feature_columns, target, label_encoders)
    metrics["probability_table"] = probability_table
    
    return metrics, pd.Series(y_pred, index=y_test.index, name="y_pred")


def _create_probability_table(df: pd.DataFrame, features: List[str], target: str, label_encoders: Dict) -> Dict:
    """Tạo bảng xác suất theo format ví dụ."""
    
    # Tính xác suất prior cho target
    target_counts = df[target].value_counts()
    total_samples = len(df)
    prior_probs = (target_counts / total_samples).to_dict()
    
    # Tạo bảng xác suất cho từng feature
    feature_tables = {}
    
    for feature in features:
        if feature in label_encoders:
            # Categorical feature - tạo bảng như ví dụ
            feature_table = pd.crosstab(df[feature], df[target], normalize='columns')
            feature_tables[feature] = {
                'type': 'categorical',
                'table': feature_table.to_dict(),
                'values': list(df[feature].unique())
            }
        else:
            # Numerical feature - tính mean và std cho mỗi class
            feature_stats = df.groupby(target)[feature].agg(['mean', 'std']).to_dict()
            feature_tables[feature] = {
                'type': 'numerical',
                'stats': feature_stats,
                'values': list(df[feature].unique())
            }
    
    return {
        'prior_probabilities': prior_probs,
        'feature_tables': feature_tables,
        'total_samples': total_samples,
        'target_classes': list(target_counts.index)
    }


def calculate_prediction_probability(sample_data: Dict, probability_table: Dict, features: List[str]) -> Dict:
    """
    Tính xác suất dự đoán cho một mẫu dữ liệu theo format ví dụ.
    
    Args:
        sample_data: Dict chứa giá trị của các features
        probability_table: Bảng xác suất đã tính
        features: Danh sách features
    
    Returns:
        Dict chứa xác suất cho mỗi class
    """
    
    prior_probs = probability_table['prior_probabilities']
    feature_tables = probability_table['feature_tables']
    
    class_probabilities = {}
    
    for class_name in probability_table['target_classes']:
        # Bắt đầu với xác suất prior
        prob = prior_probs[class_name]
        
        # Nhân với xác suất của từng feature
        for feature in features:
            if feature in sample_data:
                feature_value = sample_data[feature]
                
                if feature_tables[feature]['type'] == 'categorical':
                    # Lấy xác suất từ bảng
                    feature_table = feature_tables[feature]['table']
                    if feature_value in feature_table and class_name in feature_table[feature_value]:
                        prob *= feature_table[feature_value][class_name]
                    else:
                        prob *= 0.01  # Smoothing cho giá trị không có
                else:
                    # Numerical feature - sử dụng Gaussian distribution
                    stats = feature_tables[feature]['stats']
                    if class_name in stats['mean'] and class_name in stats['std']:
                        mean = stats['mean'][class_name]
                        std = stats['std'][class_name]
                        if std > 0:
                            # Tính xác suất từ Gaussian distribution
                            prob *= np.exp(-0.5 * ((feature_value - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        
        class_probabilities[class_name] = prob
    
    return class_probabilities


def create_customer_insights(df: pd.DataFrame, probability_table: Dict) -> Dict:
    """
    Tạo insights về khách hàng dễ hiểu.
    """
    insights = {}
    
    # Phân tích theo sản phẩm
    product_analysis = df.groupby('Item Purchased')['Will_Return'].agg(['count', 'sum', 'mean']).round(3)
    product_analysis.columns = ['Tổng_số', 'Sẽ_mua_lại', 'Tỷ_lệ_mua_lại']
    product_analysis = product_analysis.sort_values('Tỷ_lệ_mua_lại', ascending=False)
    insights['product_analysis'] = product_analysis
    
    # Phân tích theo phương thức thanh toán
    payment_analysis = df.groupby('Payment Method')['Will_Return'].agg(['count', 'sum', 'mean']).round(3)
    payment_analysis.columns = ['Tổng_số', 'Sẽ_mua_lại', 'Tỷ_lệ_mua_lại']
    payment_analysis = payment_analysis.sort_values('Tỷ_lệ_mua_lại', ascending=False)
    insights['payment_analysis'] = payment_analysis
    
    # Phân tích theo mức chi tiêu
    df_copy = df.copy()
    df_copy['Chi_tiêu_nhóm'] = pd.cut(df_copy['Purchase Amount (USD)'], 
                                     bins=[0, 50, 100, 200, 1000], 
                                     labels=['Thấp (<$50)', 'Trung bình ($50-100)', 'Cao ($100-200)', 'Rất cao (>$200)'])
    
    spending_analysis = df_copy.groupby('Chi_tiêu_nhóm', observed=True)['Will_Return'].agg(['count', 'sum', 'mean']).round(3)
    spending_analysis.columns = ['Tổng_số', 'Sẽ_mua_lại', 'Tỷ_lệ_mua_lại']
    insights['spending_analysis'] = spending_analysis
    
    # Phân tích theo rating
    df_copy['Rating_nhóm'] = pd.cut(df_copy['Review Rating'], 
                                   bins=[0, 3, 4, 5], 
                                   labels=['Thấp (<3)', 'Trung bình (3-4)', 'Cao (4-5)'])
    
    rating_analysis = df_copy.groupby('Rating_nhóm', observed=True)['Will_Return'].agg(['count', 'sum', 'mean']).round(3)
    rating_analysis.columns = ['Tổng_số', 'Sẽ_mua_lại', 'Tỷ_lệ_mua_lại']
    insights['rating_analysis'] = rating_analysis
    
    return insights


def generate_business_recommendations(insights: Dict) -> List[str]:
    """
    Tạo khuyến nghị kinh doanh dựa trên insights.
    """
    recommendations = []
    
    # Khuyến nghị về sản phẩm
    product_analysis = insights['product_analysis']
    best_product = product_analysis.index[0]
    best_rate = product_analysis.iloc[0]['Tỷ_lệ_mua_lại']
    recommendations.append(f"**Sản phẩm tốt nhất**: {best_product} có tỷ lệ mua lại {best_rate:.1%}")
    
    worst_product = product_analysis.index[-1]
    worst_rate = product_analysis.iloc[-1]['Tỷ_lệ_mua_lại']
    recommendations.append(f"**Cần cải thiện**: {worst_product} chỉ có tỷ lệ mua lại {worst_rate:.1%}")
    
    # Khuyến nghị về phương thức thanh toán
    payment_analysis = insights['payment_analysis']
    best_payment = payment_analysis.index[0]
    best_payment_rate = payment_analysis.iloc[0]['Tỷ_lệ_mua_lại']
    recommendations.append(f"Phương thức thanh toán: {best_payment} có tỷ lệ mua lại cao nhất ({best_payment_rate:.1%})")
    
    # Khuyến nghị về chiến lược
    if best_rate > 0.7:
        recommendations.append("**Chiến lược**: Tập trung phát triển sản phẩm có tỷ lệ mua lại cao")
    else:
        recommendations.append("Chiến lược: Cần cải thiện chất lượng sản phẩm và dịch vụ")
    
    # Khuyến nghị về target
    rating_analysis = insights['rating_analysis']
    high_rating_rate = rating_analysis.loc['Cao (4-5)', 'Tỷ_lệ_mua_lại'] if 'Cao (4-5)' in rating_analysis.index else 0
    recommendations.append(f"**Mục tiêu**: Tăng tỷ lệ rating cao để đạt {high_rating_rate:.1%} khách hàng mua lại")
    
    return recommendations


