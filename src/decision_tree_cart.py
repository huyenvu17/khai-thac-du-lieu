"""
Decision Tree CART Implementation - Product Restock Decision
Bài toán: Quyết định sản phẩm nên nhập/nên dừng (tương tự bài "Play Tennis" nhưng dùng Gini)

Tương tự bài "Play Tennis":
- Play Tennis: Outlook, Temperature, Humidity, Wind → Play (Yes/No) - ID3 (Entropy)
- Restock Decision: Sales_Volume, Profit_Margin, Customer_Demand, Seasonality → Restock (Yes/No) - CART (Gini)

Sự khác biệt: ID3 dùng Entropy, CART dùng Gini Impurity
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


def run_dt_cart(df: pd.DataFrame) -> Dict:
    """
    Decision Tree CART - Quyết định sản phẩm nên nhập/nên dừng
    Tương tự bài "Play Tennis" nhưng sử dụng Gini Impurity thay vì Entropy
    """
    # Tạo features mới từ dữ liệu hiện tại
    processed_df = create_restock_features(df)
    
    # Định nghĩa features và target
    feature_columns = ['Sales_Volume', 'Profit_Margin', 'Customer_Demand', 'Seasonality']
    target = 'Restock'
    
    # Lấy dữ liệu sạch
    analysis_df = processed_df[feature_columns + [target]].dropna()
    
    # Tính Gini Impurity và Information Gain
    gini_info = calculate_gini_and_gain(analysis_df, target, feature_columns)
    
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
    
    # Train model với gini (CART)
    model = DecisionTreeClassifier(
        criterion="gini", 
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
    results = analyze_restock_results(analysis_df, feature_columns)
    
    return {
        'accuracy': accuracy,
        'gini_info': gini_info,
        'decision_tree_rules': tree_rules,
        'decision_tree_figure': decision_tree_figure,
        'restock_analysis': results,
        'feature_columns': feature_columns,
        'target': target,
        'data_summary': {
            'total_samples': len(analysis_df),
            'should_restock_count': int((analysis_df[target] == 'Yes').sum()),
            'should_stop_count': int((analysis_df[target] == 'No').sum())
        }
    }


def create_restock_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo features mới cho bài toán quyết định nhập hàng
    Tương tự bài "Play Tennis" với 4 features categorical
    """
    processed_df = df.copy()
    
    # 1. Sales_Volume: Khối lượng bán (dựa trên số lượng giao dịch)
    sales_volume = df.groupby('Item Purchased').size()
    processed_df['Sales_Volume'] = processed_df['Item Purchased'].map(sales_volume)
    
    # Phân loại thành 3 mức: Low, Medium, High
    volume_percentiles = processed_df['Sales_Volume'].quantile([0.33, 0.67])
    processed_df['Sales_Volume'] = pd.cut(
        processed_df['Sales_Volume'], 
        bins=[0, volume_percentiles[0.33], volume_percentiles[0.67], float('inf')], 
        labels=['Low', 'Medium', 'High']
    )
    
    # 2. Profit_Margin: Tỷ suất lợi nhuận (dựa trên giá trị giao dịch)
    processed_df['Profit_Margin'] = pd.cut(
        processed_df['Purchase Amount (USD)'], 
        bins=[0, 1000, 3000, float('inf')], 
        labels=['Low', 'Medium', 'High']
    )
    
    # 3. Customer_Demand: Mức độ quan tâm của khách hàng (dựa trên rating)
    processed_df['Customer_Demand'] = processed_df['Review Rating'].apply(
        lambda x: 'High' if x >= 4.0 else 'Medium' if x >= 3.0 else 'Low'
    )
    
    # 4. Seasonality: Tính mùa vụ (dựa trên tháng mua)
    if 'Date Purchase' in processed_df.columns:
        processed_df['Date Purchase'] = pd.to_datetime(processed_df['Date Purchase'], format='%d-%m-%Y')
        processed_df['Month'] = processed_df['Date Purchase'].dt.month
        
        # Phân loại mùa
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        processed_df['Seasonality'] = processed_df['Month'].apply(get_season)
    else:
        # Fallback nếu không có ngày
        processed_df['Seasonality'] = 'Unknown'
    
    # 5. Target: Restock (dựa trên logic thực tế)
    # Nên nhập hàng nếu: Sales cao VÀ Rating tốt VÀ Giá trị hợp lý
    processed_df['Restock'] = (
        (processed_df['Sales_Volume'].isin(['Medium', 'High'])) &
        (processed_df['Customer_Demand'].isin(['Medium', 'High'])) &
        (processed_df['Profit_Margin'].isin(['Medium', 'High']))
    ).map({True: 'Yes', False: 'No'})
    
    return processed_df


def calculate_gini_and_gain(df: pd.DataFrame, target: str, features: List[str]) -> Dict:
    """
    Tính Gini Impurity và Information Gain cho từng feature
    CART sử dụng Gini thay vì Entropy
    """
    # Tính Gini Impurity tổng của dataset
    target_counts = df[target].value_counts()
    total_samples = len(df)
    
    # Gini = 1 - Σ(p²) với p là tỷ lệ của mỗi class
    gini_total = 1 - sum((count / total_samples) ** 2 for count in target_counts)
    
    # Tính Information Gain cho từng feature
    gain_results = {}
    
    for feature in features:
        feature_counts = df[feature].value_counts()
        weighted_gini = 0
        feature_gini_details = {}
        
        for feature_value in feature_counts.index:
            subset = df[df[feature] == feature_value]
            subset_size = len(subset)
            
            if subset_size > 0:
                target_counts_subset = subset[target].value_counts()
                
                # Tính Gini cho subset
                gini_subset = 1 - sum((count / subset_size) ** 2 for count in target_counts_subset)
                
                weight = subset_size / total_samples
                weighted_gini += weight * gini_subset
                
                feature_gini_details[feature_value] = {
                    'count': subset_size,
                    'gini': gini_subset,
                    'target_distribution': target_counts_subset.to_dict()
                }
        
        information_gain = gini_total - weighted_gini
        
        gain_results[feature] = {
            'information_gain': information_gain,
            'weighted_gini': weighted_gini,
            'gini_details': feature_gini_details,
            'feature_distribution': feature_counts.to_dict()
        }
    
    # Sắp xếp theo Information Gain giảm dần
    sorted_gains = dict(sorted(gain_results.items(), key=lambda x: x[1]['information_gain'], reverse=True))
    
    return {
        'total_gini': gini_total,
        'total_samples': total_samples,
        'target_distribution': target_counts.to_dict(),
        'feature_gains': sorted_gains
    }


def analyze_restock_results(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Phân tích kết quả quyết định nhập hàng
    """
    results = {}
    
    # Thống kê tổng quan
    total_samples = len(df)
    should_restock_count = (df['Restock'] == 'Yes').sum()
    should_stop_count = (df['Restock'] == 'No').sum()
    
    results['overview'] = {
        'total_samples': total_samples,
        'should_restock': should_restock_count,
        'should_stop': should_stop_count,
        'restock_rate': round(should_restock_count / total_samples * 100, 1)
    }
    
    # Phân tích theo từng feature
    for feature in features:
        feature_analysis = {}
        
        for value in df[feature].unique():
            subset = df[df[feature] == value]
            subset_size = len(subset)
            should_restock_count = (subset['Restock'] == 'Yes').sum()
            restock_rate = round(should_restock_count / subset_size * 100, 1) if subset_size > 0 else 0
            
            feature_analysis[value] = {
                'count': subset_size,
                'should_restock': should_restock_count,
                'restock_rate': restock_rate
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
                 class_names=['Không nhập', 'Nên nhập'],
                 filled=True,
                 rounded=True,
                 fontsize=9,
                 max_depth=4)
        
        # Thiết lập title và layout
        plt.title("Cây Quyết Định CART - Quyết Định Sản Phẩm Nên Nhập/Nên Dừng", 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Tối ưu layout
        plt.tight_layout()
        
        return plt.gcf()
        
    except Exception as e:
        print(f"Lỗi khi tạo sơ đồ cây quyết định: {e}")
        return None


def predict_restock_sample(model, sales_volume: str, profit_margin: str, 
                          customer_demand: str, seasonality: str) -> Dict:
    """
    Dự đoán mẫu cho một sản phẩm cụ thể
    """
    # Mapping values
    feature_mapping = {
        'Sales_Volume': {'Low': 0, 'Medium': 1, 'High': 2},
        'Profit_Margin': {'Low': 0, 'Medium': 1, 'High': 2},
        'Customer_Demand': {'Low': 0, 'Medium': 1, 'High': 2},
        'Seasonality': {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3, 'Unknown': 4}
    }
    
    # Tạo input vector
    input_vector = [
        feature_mapping['Sales_Volume'][sales_volume],
        feature_mapping['Profit_Margin'][profit_margin],
        feature_mapping['Customer_Demand'][customer_demand],
        feature_mapping['Seasonality'][seasonality]
    ]
    
    # Dự đoán
    prediction = model.predict([input_vector])[0]
    probability = model.predict_proba([input_vector])[0]
    
    return {
        'prediction': 'Nên nhập hàng' if prediction == 'Yes' else 'Không nên nhập hàng',
        'confidence': round(max(probability) * 100, 1),
        'probabilities': {
            'Không nhập': round(probability[0] * 100, 1),
            'Nên nhập': round(probability[1] * 100, 1)
        }
    }