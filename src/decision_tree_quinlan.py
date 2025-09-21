"""
Decision Tree Quinlan (C4.5) Implementation - Seasonal Trend Prediction
Bài toán: Dự đoán xu hướng mùa vụ (tương tự bài "Play Tennis" nhưng dùng C4.5)

Tương tự bài "Play Tennis":
- Play Tennis: Outlook, Temperature, Humidity, Wind → Play (Yes/No) - ID3 (Entropy)
- Purchase Decision: Item_Type, Price_Range, Payment_Preference, Customer_Type → Will_Buy (Yes/No) - CART (Gini)
- Seasonal Trend: Product_Category, Price_Level, Customer_Segment, Time_Period → Seasonal_Trend (High/Low) - C4.5 (Entropy + Gain Ratio)

Sự khác biệt: C4.5 sử dụng Gain Ratio thay vì Information Gain để tránh bias với features có nhiều giá trị
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def run_dt_quinlan(df: pd.DataFrame) -> Dict:
    """
    Decision Tree Quinlan (C4.5) - Dự đoán xu hướng mùa vụ
    Tương tự bài "Play Tennis" nhưng sử dụng Gain Ratio thay vì Information Gain
    """
    # Tạo features mới từ dữ liệu hiện tại
    processed_df = create_seasonal_features(df)
    
    # Định nghĩa features và target
    feature_columns = ['Product_Category', 'Price_Level', 'Customer_Segment', 'Time_Period']
    target = 'Seasonal_Trend'
    
    # Lấy dữ liệu sạch
    analysis_df = processed_df[feature_columns + [target]].dropna()
    
    # Tính Entropy và Gain Ratio
    gain_ratio_info = calculate_entropy_and_gain_ratio(analysis_df, target, feature_columns)
    
    # Xử lý categorical features
    processed_df_final = analysis_df.copy()
    label_encoders = {}
    
    for col in feature_columns:
        le = LabelEncoder()
        processed_df_final[col] = le.fit_transform(analysis_df[col].astype(str))
        label_encoders[col] = le
    
    # Chia train/test
    X = processed_df_final[feature_columns]
    y = processed_df_final[target]
    
    # Train model với entropy (C4.5)
    model = DecisionTreeClassifier(
        criterion="entropy", 
        max_depth=4,  # Đủ sâu để thể hiện logic
        min_samples_split=2,
        random_state=42
    )
    model.fit(X, y)
    
    # Tính accuracy
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    # Tạo sơ đồ cây quyết định dạng text
    tree_rules = export_text(model, feature_names=feature_columns, max_depth=4)
    
    # Tạo sơ đồ cây quyết định trực quan
    decision_tree_figure = create_visual_decision_tree(model, feature_columns)
    
    # Phân tích kết quả
    results = analyze_seasonal_results(analysis_df, feature_columns)
    
    return {
        'accuracy': accuracy,
        'gain_ratio_info': gain_ratio_info,
        'decision_tree_rules': tree_rules,
        'decision_tree_figure': decision_tree_figure,
        'seasonal_analysis': results,
        'feature_columns': feature_columns,
        'target': target,
        'data_summary': {
            'total_samples': len(analysis_df),
            'high_trend_count': int((analysis_df[target] == 'High').sum()),
            'low_trend_count': int((analysis_df[target] == 'Low').sum())
        }
    }


def create_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo features mới cho bài toán dự đoán xu hướng mùa vụ
    Tương tự bài "Play Tennis" với 4 features categorical
    """
    processed_df = df.copy()
    
    # 1. Product_Category: Phân loại sản phẩm
    product_category_mapping = {
        'Dress': 'Clothing', 'Shirt': 'Clothing', 'Jacket': 'Clothing', 'Jeans': 'Clothing',
        'Tunic': 'Clothing', 'Tank Top': 'Clothing', 'Leggings': 'Clothing', 'Trousers': 'Clothing',
        'Polo Shirt': 'Clothing', 'Camisole': 'Clothing', 'Jumpsuit': 'Clothing', 'Skirt': 'Clothing',
        'Handbag': 'Accessories', 'Wallet': 'Accessories', 'Gloves': 'Accessories', 'Bowtie': 'Accessories',
        'Hat': 'Accessories', 'Onesie': 'Accessories', 'Pajamas': 'Accessories', 'Poncho': 'Accessories',
        'Raincoat': 'Accessories', 'Trench Coat': 'Accessories',
        'Shoes': 'Footwear', 'Slippers': 'Footwear', 'Sneakers': 'Footwear', 'Loafers': 'Footwear',
        'Sandals': 'Footwear'
    }
    
    processed_df['Product_Category'] = processed_df['Item Purchased'].map(product_category_mapping).fillna('Accessories')
    
    # 2. Price_Level: Mức giá (dựa trên giá trị giao dịch)
    processed_df['Price_Level'] = pd.cut(
        processed_df['Purchase Amount (USD)'], 
        bins=[0, 1000, 3000, float('inf')], 
        labels=['Budget', 'Mid-Range', 'Premium']
    )
    
    # 3. Customer_Segment: Phân khúc khách hàng (dựa trên Will_Return và Rating)
    def get_customer_segment(row):
        if row['Will_Return'] == 1 and row['Review Rating'] >= 4.0:
            return 'Loyal'
        elif row['Review Rating'] >= 4.0:
            return 'Satisfied'
        elif row['Will_Return'] == 1:
            return 'Returning'
        else:
            return 'New'
    
    processed_df['Customer_Segment'] = processed_df.apply(get_customer_segment, axis=1)
    
    # 4. Time_Period: Thời kỳ (dựa trên tháng mua)
    if 'Date Purchase' in processed_df.columns:
        processed_df['Date Purchase'] = pd.to_datetime(processed_df['Date Purchase'], format='%d-%m-%Y')
        processed_df['Month'] = processed_df['Date Purchase'].dt.month
        
        # Phân loại thời kỳ
        def get_time_period(month):
            if month in [12, 1, 2]:
                return 'Holiday_Season'
            elif month in [3, 4, 5]:
                return 'Spring_Season'
            elif month in [6, 7, 8]:
                return 'Summer_Season'
            else:
                return 'Fall_Season'
        
        processed_df['Time_Period'] = processed_df['Month'].apply(get_time_period)
    else:
        # Fallback nếu không có ngày
        processed_df['Time_Period'] = 'Unknown'
    
    # 5. Target: Seasonal_Trend (dựa trên logic thực tế)
    # Xu hướng mùa vụ cao nếu: Rating tốt VÀ giá trị cao VÀ thời kỳ phù hợp
    def get_seasonal_trend(row):
        # Holiday season và Summer season thường có xu hướng cao
        high_season_periods = ['Holiday_Season', 'Summer_Season']
        
        if (row['Review Rating'] >= 4.0 and 
            row['Price_Level'] in ['Mid-Range', 'Premium'] and
            row['Time_Period'] in high_season_periods):
            return 'High'
        else:
            return 'Low'
    
    processed_df['Seasonal_Trend'] = processed_df.apply(get_seasonal_trend, axis=1)
    
    return processed_df


def calculate_entropy_and_gain_ratio(df: pd.DataFrame, target: str, features: List[str]) -> Dict:
    """
    Tính Entropy và Gain Ratio cho từng feature
    C4.5 sử dụng Gain Ratio thay vì Information Gain để tránh bias
    """
    # Tính Entropy tổng của dataset
    target_counts = df[target].value_counts()
    total_samples = len(df)
    entropy_total = 0
    
    for count in target_counts:
        p = count / total_samples
        if p > 0:
            entropy_total -= p * np.log2(p)
    
    # Tính Gain Ratio cho từng feature
    gain_ratio_results = {}
    
    for feature in features:
        feature_counts = df[feature].value_counts()
        weighted_entropy = 0
        split_info = 0
        feature_entropy_details = {}
        
        for feature_value in feature_counts.index:
            subset = df[df[feature] == feature_value]
            subset_size = len(subset)
            
            if subset_size > 0:
                target_counts_subset = subset[target].value_counts()
                entropy_subset = 0
                
                for count in target_counts_subset:
                    p = count / subset_size
                    if p > 0:
                        entropy_subset -= p * np.log2(p)
                
                weight = subset_size / total_samples
                weighted_entropy += weight * entropy_subset
                
                # Tính Split Information
                p_split = subset_size / total_samples
                if p_split > 0:
                    split_info -= p_split * np.log2(p_split)
                
                feature_entropy_details[feature_value] = {
                    'count': subset_size,
                    'entropy': entropy_subset,
                    'target_distribution': target_counts_subset.to_dict()
                }
        
        information_gain = entropy_total - weighted_entropy
        gain_ratio = information_gain / split_info if split_info > 0 else 0
        
        gain_ratio_results[feature] = {
            'information_gain': information_gain,
            'split_information': split_info,
            'gain_ratio': gain_ratio,
            'weighted_entropy': weighted_entropy,
            'entropy_details': feature_entropy_details,
            'feature_distribution': feature_counts.to_dict()
        }
    
    # Sắp xếp theo Gain Ratio giảm dần
    sorted_gains = dict(sorted(gain_ratio_results.items(), key=lambda x: x[1]['gain_ratio'], reverse=True))
    
    return {
        'total_entropy': entropy_total,
        'total_samples': total_samples,
        'target_distribution': target_counts.to_dict(),
        'feature_gains': sorted_gains
    }


def analyze_seasonal_results(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Phân tích kết quả dự đoán xu hướng mùa vụ
    """
    results = {}
    
    # Thống kê tổng quan
    total_samples = len(df)
    high_trend_count = (df['Seasonal_Trend'] == 'High').sum()
    low_trend_count = (df['Seasonal_Trend'] == 'Low').sum()
    
    results['overview'] = {
        'total_samples': total_samples,
        'high_trend': high_trend_count,
        'low_trend': low_trend_count,
        'high_trend_rate': round(high_trend_count / total_samples * 100, 1)
    }
    
    # Phân tích theo từng feature
    for feature in features:
        feature_analysis = {}
        
        for value in df[feature].unique():
            subset = df[df[feature] == value]
            subset_size = len(subset)
            high_trend_count = (subset['Seasonal_Trend'] == 'High').sum()
            high_trend_rate = round(high_trend_count / subset_size * 100, 1) if subset_size > 0 else 0
            
            feature_analysis[value] = {
                'count': subset_size,
                'high_trend': high_trend_count,
                'high_trend_rate': high_trend_rate
            }
        
        results[feature] = feature_analysis
    
    return results


def create_visual_decision_tree(model, feature_names):
    """
    Tạo sơ đồ cây quyết định trực quan
    """
    try:
        # Tạo figure với kích thước phù hợp
        plt.figure(figsize=(15, 10))
        
        # Vẽ cây quyết định
        plot_tree(model, 
                 feature_names=feature_names,
                 class_names=['Low Trend', 'High Trend'],
                 filled=True,
                 rounded=True,
                 fontsize=9,
                 max_depth=4)
        
        # Thiết lập title và layout
        plt.title("Cây Quyết Định C4.5 - Dự Đoán Xu Hướng Mùa Vụ", 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Tối ưu layout
        plt.tight_layout()
        
        return plt.gcf()
        
    except Exception as e:
        print(f"Lỗi khi tạo sơ đồ cây quyết định: {e}")
        return None


def predict_seasonal_trend_sample(model, product_category: str, price_level: str, 
                                 customer_segment: str, time_period: str) -> Dict:
    """
    Dự đoán mẫu cho một sản phẩm cụ thể
    """
    # Mapping values
    feature_mapping = {
        'Product_Category': {'Clothing': 0, 'Accessories': 1, 'Footwear': 2},
        'Price_Level': {'Budget': 0, 'Mid-Range': 1, 'Premium': 2},
        'Customer_Segment': {'New': 0, 'Returning': 1, 'Satisfied': 2, 'Loyal': 3},
        'Time_Period': {'Holiday_Season': 0, 'Spring_Season': 1, 'Summer_Season': 2, 'Fall_Season': 3, 'Unknown': 4}
    }
    
    # Tạo input vector
    input_vector = [
        feature_mapping['Product_Category'][product_category],
        feature_mapping['Price_Level'][price_level],
        feature_mapping['Customer_Segment'][customer_segment],
        feature_mapping['Time_Period'][time_period]
    ]
    
    # Dự đoán
    prediction = model.predict([input_vector])[0]
    probability = model.predict_proba([input_vector])[0]
    
    return {
        'prediction': 'High Seasonal Trend' if prediction == 'High' else 'Low Seasonal Trend',
        'confidence': round(max(probability) * 100, 1),
        'probabilities': {
            'Low Trend': round(probability[0] * 100, 1),
            'High Trend': round(probability[1] * 100, 1)
        }
    }