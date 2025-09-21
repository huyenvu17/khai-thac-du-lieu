"""
Decision Tree ID3 Implementation - Purchase Decision Prediction
Bài toán: Dự đoán khách hàng có mua hàng hay không (tương tự bài "Play Tennis" kinh điển)

Tương tự bài "Play Tennis":
- Play Tennis: Outlook, Temperature, Humidity, Wind → Play (Yes/No)
- Purchase Decision: Item_Type, Price_Range, Payment_Preference, Customer_Type → Will_Buy (Yes/No)
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


def run_dt_id3(df: pd.DataFrame) -> Dict:
    """
    Decision Tree ID3 - Dự đoán khách hàng có mua hàng hay không
    Tương tự bài "Play Tennis" kinh điển
    """
    # Tạo features mới từ dữ liệu hiện tại
    processed_df = create_purchase_features(df)
    
    # Định nghĩa features và target
    feature_columns = ['Item_Type', 'Price_Range', 'Payment_Preference', 'Customer_Type']
    target = 'Will_Buy'
    
    # Lấy dữ liệu sạch
    analysis_df = processed_df[feature_columns + [target]].dropna()
    
    # Tính Entropy và Information Gain
    entropy_info = calculate_entropy_and_gain(analysis_df, target, feature_columns)
    
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
    
    # Train model với entropy (ID3)
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
    results = analyze_purchase_decisions(analysis_df, feature_columns)
    
    return {
        'accuracy': accuracy,
        'entropy_info': entropy_info,
        'decision_tree_rules': tree_rules,
        'decision_tree_figure': decision_tree_figure,
        'purchase_analysis': results,
        'feature_columns': feature_columns,
        'target': target,
        'data_summary': {
            'total_samples': len(analysis_df),
            'will_buy_count': int(analysis_df[target].sum()),
            'will_not_buy_count': int(len(analysis_df) - analysis_df[target].sum())
        }
    }


def create_purchase_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo features mới cho bài toán dự đoán mua hàng
    Tương tự bài "Play Tennis" với 4 features categorical
    """
    processed_df = df.copy()
    
    # 1. Item_Type: Phân loại sản phẩm
    item_type_mapping = {
        'Dress': 'Clothing', 'Shirt': 'Clothing', 'Jacket': 'Clothing', 'Jeans': 'Clothing',
        'Tunic': 'Clothing', 'Tank Top': 'Clothing', 'Leggings': 'Clothing', 'Trousers': 'Clothing',
        'Polo Shirt': 'Clothing', 'Camisole': 'Clothing', 'Jumpsuit': 'Clothing', 'Skirt': 'Clothing',
        'Handbag': 'Accessories', 'Wallet': 'Accessories', 'Gloves': 'Accessories', 'Bowtie': 'Accessories',
        'Hat': 'Accessories', 'Onesie': 'Accessories', 'Pajamas': 'Accessories', 'Poncho': 'Accessories',
        'Raincoat': 'Accessories', 'Trench Coat': 'Accessories',
        'Shoes': 'Footwear', 'Slippers': 'Footwear', 'Sneakers': 'Footwear', 'Loafers': 'Footwear',
        'Sandals': 'Footwear'
    }
    
    processed_df['Item_Type'] = processed_df['Item Purchased'].map(item_type_mapping).fillna('Accessories')
    
    # 2. Price_Range: Phân loại giá
    processed_df['Price_Range'] = pd.cut(
        processed_df['Purchase Amount (USD)'], 
        bins=[0, 500, 2000, float('inf')], 
        labels=['Low', 'Medium', 'High']
    )
    
    # 3. Payment_Preference: Phương thức thanh toán
    processed_df['Payment_Preference'] = processed_df['Payment Method']
    
    # 4. Customer_Type: Loại khách hàng (dựa trên Will_Return)
    processed_df['Customer_Type'] = processed_df['Will_Return'].map({1: 'Returning', 0: 'New'})
    
    # 5. Target: Will_Buy (dựa trên logic thực tế)
    # Khách hàng có khả năng mua cao nếu: Rating tốt VÀ giá trị hợp lý
    processed_df['Will_Buy'] = (
        (processed_df['Review Rating'] >= 3.5) &  # Rating trung bình trở lên
        (processed_df['Purchase Amount (USD)'] >= 100)  # Giá trị tối thiểu
    ).astype(int)
    
    return processed_df


def calculate_entropy_and_gain(df: pd.DataFrame, target: str, features: List[str]) -> Dict:
    """
    Tính Entropy và Information Gain cho từng feature
    """
    # Tính Entropy tổng của dataset
    target_counts = df[target].value_counts()
    total_samples = len(df)
    entropy_total = 0
    
    for count in target_counts:
        p = count / total_samples
        if p > 0:
            entropy_total -= p * np.log2(p)
    
    # Tính Information Gain cho từng feature
    gain_results = {}
    
    for feature in features:
        feature_counts = df[feature].value_counts()
        weighted_entropy = 0
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
                
                feature_entropy_details[feature_value] = {
                    'count': subset_size,
                    'entropy': entropy_subset,
                    'target_distribution': target_counts_subset.to_dict()
                }
        
        information_gain = entropy_total - weighted_entropy
        
        gain_results[feature] = {
            'information_gain': information_gain,
            'weighted_entropy': weighted_entropy,
            'entropy_details': feature_entropy_details,
            'feature_distribution': feature_counts.to_dict()
        }
    
    # Sắp xếp theo Information Gain giảm dần
    sorted_gains = dict(sorted(gain_results.items(), key=lambda x: x[1]['information_gain'], reverse=True))
    
    return {
        'total_entropy': entropy_total,
        'total_samples': total_samples,
        'target_distribution': target_counts.to_dict(),
        'feature_gains': sorted_gains
    }


def analyze_purchase_decisions(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Phân tích kết quả dự đoán mua hàng
    """
    results = {}
    
    # Thống kê tổng quan
    total_samples = len(df)
    will_buy_count = df['Will_Buy'].sum()
    will_not_buy_count = total_samples - will_buy_count
    
    results['overview'] = {
        'total_samples': total_samples,
        'will_buy': will_buy_count,
        'will_not_buy': will_not_buy_count,
        'buy_rate': round(will_buy_count / total_samples * 100, 1)
    }
    
    # Phân tích theo từng feature
    for feature in features:
        feature_analysis = {}
        
        for value in df[feature].unique():
            subset = df[df[feature] == value]
            subset_size = len(subset)
            buy_count = subset['Will_Buy'].sum()
            buy_rate = round(buy_count / subset_size * 100, 1) if subset_size > 0 else 0
            
            feature_analysis[value] = {
                'count': subset_size,
                'will_buy': buy_count,
                'buy_rate': buy_rate
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
                 class_names=['Không mua', 'Sẽ mua'],
                 filled=True,
                 rounded=True,
                 fontsize=9,
                 max_depth=4)
        
        # Thiết lập title và layout
        plt.title("Cây Quyết Định ID3 - Dự Đoán Khách Hàng Có Mua Hàng Hay Không", 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Tối ưu layout
        plt.tight_layout()
        
        return plt.gcf()
        
    except Exception as e:
        print(f"Lỗi khi tạo sơ đồ cây quyết định: {e}")
        return None


def predict_purchase_sample(model, item_type: str, price_range: str, 
                          payment_preference: str, customer_type: str) -> Dict:
    """
    Dự đoán mẫu cho một khách hàng cụ thể
    """
    # Mapping values
    feature_mapping = {
        'Item_Type': {'Clothing': 0, 'Accessories': 1, 'Footwear': 2},
        'Price_Range': {'Low': 0, 'Medium': 1, 'High': 2},
        'Payment_Preference': {'Cash': 0, 'Credit Card': 1, 'Debit Card': 2},
        'Customer_Type': {'New': 0, 'Returning': 1}
    }
    
    # Tạo input vector
    input_vector = [
        feature_mapping['Item_Type'][item_type],
        feature_mapping['Price_Range'][price_range],
        feature_mapping['Payment_Preference'][payment_preference],
        feature_mapping['Customer_Type'][customer_type]
    ]
    
    # Dự đoán
    prediction = model.predict([input_vector])[0]
    probability = model.predict_proba([input_vector])[0]
    
    return {
        'prediction': 'Sẽ mua' if prediction == 1 else 'Không mua',
        'confidence': round(max(probability) * 100, 1),
        'probabilities': {
            'Không mua': round(probability[0] * 100, 1),
            'Sẽ mua': round(probability[1] * 100, 1)
        }
    }