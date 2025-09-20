from typing import List
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def analyze_review_factors(df: pd.DataFrame) -> dict:
    """Phân tích các yếu tố ảnh hưởng đến đánh giá tốt của khách hàng.
    
    Xác định những yếu tố cốt lõi và tối thiểu có ảnh hưởng mạnh nhất 
    đến việc khách hàng để lại một đánh giá tốt.
    
    Args:
        df: DataFrame chứa dữ liệu bán hàng
        
    Returns:
        dict: Kết quả phân tích với các yếu tố quan trọng và insights
    """
    # Chuẩn bị dữ liệu - loại bỏ missing values trong Review Rating
    data = df.dropna(subset=['Review Rating']).copy()
    
    # Định nghĩa đánh giá tốt: Rating >= 4.0
    data['High_Rating'] = (data['Review Rating'] >= 4.0).astype(int)
    
    # Chọn các yếu tố cốt lõi và tối thiểu
    core_factors = []
    
    # 1. Loại sản phẩm - yếu tố cốt lõi
    if 'Item Purchased' in data.columns:
        core_factors.append('Item Purchased')
    
    # 2. Phương thức thanh toán - yếu tố cốt lõi  
    if 'Payment Method' in data.columns:
        core_factors.append('Payment Method')
    
    # 3. Giá trị giao dịch - yếu tố quan trọng
    if 'Purchase Amount (USD)' in data.columns:
        core_factors.append('Purchase Amount (USD)')
    
    # Tạo dữ liệu phân tích với các cột cần thiết
    analysis_cols = core_factors + ['High_Rating', 'Review Rating']
    analysis_data = data[analysis_cols].copy()
    
    # Xử lý missing values triệt để - thay thế tất cả NaN
    analysis_data = analysis_data.fillna({
        'Item Purchased': 'Unknown',
        'Payment Method': 'Unknown', 
        'Purchase Amount (USD)': analysis_data['Purchase Amount (USD)'].median() if analysis_data['Purchase Amount (USD)'].notna().any() else 0,
        'High_Rating': 0,
        'Review Rating': 0
    })
    
    # Kiểm tra cuối cùng - đảm bảo không còn NaN
    assert not analysis_data.isnull().any().any(), f"Dữ liệu vẫn còn NaN: {analysis_data.isnull().sum()}"
    
    # Tìm yếu tố quan trọng nhất bằng Mutual Information
    important_factors = _find_core_factors(analysis_data, core_factors)
    
    # Tạo insights chi tiết
    insights = _generate_detailed_insights(analysis_data, important_factors)
    
    return insights


def _find_core_factors(data: pd.DataFrame, core_factors: List[str]) -> List[str]:
    """Tìm các yếu tố cốt lõi có ảnh hưởng mạnh nhất đến đánh giá tốt."""
    features = data[core_factors].copy()
    target = data['High_Rating']
    
    # Dữ liệu đã được xử lý sạch từ function chính, chỉ cần kiểm tra
    assert not features.isnull().any().any(), f"Features vẫn còn NaN: {features.isnull().sum()}"
    assert not target.isnull().any(), f"Target vẫn còn NaN: {target.isnull().sum()}"
    
    # Mã hóa categorical features
    for col in features.columns:
        if features[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
    
    # Tính Mutual Information
    mi_scores = mutual_info_classif(features, target, random_state=42)
    
    # Sắp xếp theo độ quan trọng
    factor_importance = pd.Series(mi_scores, index=core_factors)
    factor_importance = factor_importance.sort_values(ascending=False)
    
    return factor_importance.index.tolist()


def _generate_detailed_insights(data: pd.DataFrame, important_factors: List[str]) -> dict:
    """Tạo insights chi tiết về các yếu tố ảnh hưởng đến đánh giá tốt."""
    
    # Thống kê tổng quan
    total_reviews = len(data)
    high_rating_count = data['High_Rating'].sum()
    high_rating_rate = high_rating_count / total_reviews
    avg_rating = data['Review Rating'].mean()
    
    # Phân tích từng yếu tố quan trọng
    factor_analysis = {}
    
    for factor in important_factors:
        if data[factor].dtype in ['object', 'category']:
            # Yếu tố categorical (Item Purchased, Payment Method)
            factor_stats = data.groupby(factor)['High_Rating'].agg(['count', 'mean']).round(3)
            factor_stats.columns = ['Total_Reviews', 'High_Rating_Rate']
            
            # Kiểm tra nếu có dữ liệu
            if len(factor_stats) > 0:
                # Tìm loại có tỷ lệ đánh giá tốt cao nhất
                best_category_idx = factor_stats['High_Rating_Rate'].idxmax()
                best_category = factor_stats.loc[best_category_idx]
                
                # Tạo details dạng đơn giản
                details_dict = {}
                for category, stats in factor_stats.iterrows():
                    details_dict[category] = {
                        'Total_Reviews': int(stats['Total_Reviews']),
                        'High_Rating_Rate': float(stats['High_Rating_Rate'])
                    }
                
                factor_analysis[factor] = {
                    'type': 'categorical',
                    'best_category': best_category_idx,
                    'best_rate': float(best_category['High_Rating_Rate']),
                    'details': details_dict
                }
            else:
                # Trường hợp không có dữ liệu
                factor_analysis[factor] = {
                    'type': 'categorical',
                    'best_category': 'N/A',
                    'best_rate': 0.0,
                    'details': {}
                }
            
        else:
            # Yếu tố numerical (Purchase Amount)
            high_rating_data = data[data['High_Rating'] == 1][factor]
            low_rating_data = data[data['High_Rating'] == 0][factor]
            
            avg_high = high_rating_data.mean()
            avg_low = low_rating_data.mean()
            difference = avg_high - avg_low
            
            factor_analysis[factor] = {
                'type': 'numerical',
                'high_rating_avg': round(avg_high, 2),
                'low_rating_avg': round(avg_low, 2),
                'difference': round(difference, 2),
                'impact': 'positive' if difference > 0 else 'negative'
            }
    
    # Tạo khuyến nghị
    recommendations = _generate_recommendations(factor_analysis)
    
    return {
        'summary': {
            'total_reviews': total_reviews,
            'high_rating_count': high_rating_count,
            'high_rating_rate': round(high_rating_rate, 3),
            'avg_rating': round(avg_rating, 2)
        },
        'important_factors': important_factors,
        'factor_analysis': factor_analysis,
        'recommendations': recommendations
    }


def _generate_recommendations(factor_analysis: dict) -> List[str]:
    """Tạo khuyến nghị dựa trên phân tích các yếu tố."""
    recommendations = []
    
    for factor, analysis in factor_analysis.items():
        if analysis['type'] == 'categorical':
            best_cat = analysis['best_category']
            best_rate = analysis['best_rate']
            
            if factor == 'Item Purchased':
                recommendations.append(f"🎯 **Tập trung vào sản phẩm '{best_cat}'** - có tỷ lệ đánh giá tốt cao nhất ({best_rate:.1%})")
            elif factor == 'Payment Method':
                recommendations.append(f"💳 **Ưu tiên phương thức thanh toán '{best_cat}'** - khách hàng hài lòng nhất ({best_rate:.1%})")
                
        elif analysis['type'] == 'numerical':
            if analysis['impact'] == 'positive':
                recommendations.append(f"💰 **Tăng giá trị giao dịch** - khách hàng mua nhiều hơn có xu hướng đánh giá tốt hơn (+{analysis['difference']:.2f})")
            else:
                recommendations.append(f"📉 **Kiểm tra giá trị giao dịch** - cần tối ưu để cải thiện đánh giá")
    
    # Khuyến nghị tổng quát
    recommendations.extend([
        "📊 **Theo dõi thường xuyên** các yếu tố quan trọng để duy trì chất lượng",
        "🎯 **Tập trung marketing** vào các yếu tố có tác động mạnh nhất",
        "📈 **Đo lường hiệu quả** sau khi áp dụng các khuyến nghị"
    ])
    
    return recommendations


# Backward compatibility
def reduce_attributes(df: pd.DataFrame, target: str, max_features: int = 5) -> List[str]:
    """Wrapper function để tương thích với code cũ."""
    if target == 'High_Revenue':
        # Logic cũ cho High_Revenue
        y = df[target]
        X = df.drop(columns=[target])
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) > 0:
            corrs = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
            return corrs.head(max_features).index.tolist()
        return X.columns[:max_features].tolist()
    else:
        # Sử dụng logic mới cho Review Rating
        insights = analyze_review_factors(df)
        return insights['important_factors']


