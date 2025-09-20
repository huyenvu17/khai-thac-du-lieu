import streamlit as st
import pandas as pd
from src.utils import load_csv
from src.kmeans import run_kmeans, plot_clusters, plot_elbow_method, plot_cluster_stats
from src.naive_bayes import run_nb
from src.decision_tree_cart import run_dt_cart
from src.decision_tree_id3 import run_dt_id3
from src.decision_tree_quinlan import run_dt_quinlan
from src.apriori import run_apriori
from src.roughset import reduce_attributes


st.set_page_config(page_title="Ứng Dụng Khai Thác Dữ Liệu Trong Lĩnh Vực Bán Hàng", layout="wide")


@st.cache_data
def load_fashion_retail_data() -> pd.DataFrame:
    try:
        return load_csv("data/datasets/Fashion_Retail_Sales_one.csv")
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_default_customers() -> pd.DataFrame:
    try:
        return load_csv("data/datasets/customers.csv")
    except Exception:
        return pd.DataFrame()


def sidebar_controls():
    st.sidebar.header("Chọn thuật toán")
    algo = st.sidebar.selectbox(
        "Thuật toán",
        [
            "Apriori",
            "K-means", 
            "Naive Bayes",
            "Decision-Tree-CART",
            "Decision-Tree-ID3",
            "Decision-Tree-Quinlan",
            "Rough Set",
        ],
    )

    st.sidebar.header("Dữ liệu")
    data_choice = st.sidebar.radio("Nguồn dữ liệu", ["Mặc định", "Tải CSV"])
    uploaded = None
    if data_choice == "Tải CSV":
        uploaded = st.sidebar.file_uploader("Chọn file CSV", type=["csv"]) 

    return algo, uploaded


def main():
    st.title("Ứng Dụng Khai Thác Dữ Liệu Trong Lĩnh Vực Bán Hàng")
    st.write("Chọn thuật toán và tham số ở sidebar. Kết quả hiển thị ở đây.")

    algo, uploaded = sidebar_controls()

    # Load data theo thuật toán
    data_df = pd.DataFrame()
    if uploaded is not None:
        data_df = pd.read_csv(uploaded)
    else:
        data_df = load_fashion_retail_data()
    
    if data_df.empty:
        st.warning("Không tìm thấy dữ liệu Fashion Retail Sales. Hãy tải CSV hoặc thêm vào data/datasets/Fashion_Retail_Sales.csv")
        return

    st.subheader("Dữ liệu sử dụng")
    with st.expander("Dữ liệu Fashion Retail Sales", expanded=False):
        st.dataframe(data_df)

    # Hiển thị dữ liệu đầu vào cho thuật toán được chọn
    if algo == "K-means":
        # Chuẩn bị dữ liệu cho K-means phân nhóm khách hàng
        if 'Purchase Amount (USD)' in data_df.columns and 'Review Rating' in data_df.columns:
            # Xử lý missing values
            kmeans_data = data_df.dropna(subset=['Purchase Amount (USD)', 'Review Rating']).copy()
            
            st.write("**Dữ liệu đầu vào (K-means):**")
            kmeans_input = kmeans_data[['Customer Reference ID', 'Purchase Amount (USD)', 'Review Rating']].copy()
            st.dataframe(kmeans_input.head(20))
        else:
            st.error("Dữ liệu không có cột 'Purchase Amount (USD)' hoặc 'Review Rating'")





    # Phần chạy thuật toán K-means
    if algo == "K-means":
        st.subheader("Phân Nhóm Khách Hàng - K-means")
        st.write("**Mục tiêu:** Phân nhóm khách hàng theo mức chi tiêu và độ hài lòng")
        
        # Chuẩn bị dữ liệu cho K-means phân nhóm khách hàng
        if 'Purchase Amount (USD)' in data_df.columns and 'Review Rating' in data_df.columns:
            # Xử lý missing values
            kmeans_data = data_df.dropna(subset=['Purchase Amount (USD)', 'Review Rating']).copy()
            
            k = st.slider("Số nhóm khách hàng (k)", 2, 6, 4, help="Số nhóm khách hàng muốn phân chia")

            if st.button("Chạy K-means"):
                # Chạy K-means với dữ liệu đã chuẩn bị
                labels, inertia, cluster_info = run_kmeans(kmeans_data, k=k)
                
                st.success(f"✅ Đã phân nhóm {len(kmeans_data)} khách hàng thành {k} nhóm (Inertia: {inertia:.2f})")

                # Biểu đồ elbow method
                st.subheader("Số Nhóm Khách Hàng Tối Ưu (Theo phương pháp Elbow)")
                fig_elbow = plot_elbow_method(kmeans_data, max_k=8, highlight_k=k)
                st.pyplot(fig_elbow)
                
                # Hiển thị kết quả phân cụm
                result = kmeans_data.copy()
                result["cluster"] = labels
                
                # Hiển thị thông tin chi tiết về các nhóm
                st.subheader("Thông Tin Chi Tiết Các Nhóm Khách Hàng")
                for cluster_id in sorted(cluster_info.keys()):
                    info = cluster_info[cluster_id]
                    with st.expander(f"Nhóm {cluster_id}: {info['type']}", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Số lượng", f"{info['count']} khách hàng")
                        with col2:
                            st.metric("Chi tiêu TB", f"${info['avg_amount']}")
                        with col3:
                            st.metric("Đánh giá TB", f"{info['avg_rating']}/5.0")
                        with col4:
                            st.metric("Tỷ lệ", f"{info['percentage']}%")
                

                # Biểu đồ phân cụm
                fig_clusters = plot_clusters(kmeans_data, labels, k)
                st.pyplot(fig_clusters)



                
                # Biểu đồ thống kê
                st.subheader("Thống Kê Chi Tiết")
                fig_stats = plot_cluster_stats(kmeans_data, labels, cluster_info)
                st.pyplot(fig_stats)
                
                # Khuyến nghị kinh doanh dạng bảng
                st.subheader("Khuyến Nghị Chiến Lược Cho Từng Nhóm Khách Hàng")
                
                # Tạo DataFrame cho khuyến nghị
                strategy_data = []
                for cluster_id in sorted(cluster_info.keys()):
                    info = cluster_info[cluster_id]
                    
                    if "VIP" in info['type']:
                        xep_loai = "Ưu tiên cao nhất"
                        chien_luoc = "Dịch vụ đặc biệt, sản phẩm cao cấp, chương trình VIP"
                    elif "Trung thành" in info['type']:
                        xep_loai = "Khách hàng trung thành"
                        chien_luoc = "Giữ chân, tăng giá trị đơn hàng, chương trình khuyến mãi"
                    elif "chưa hài lòng" in info['type']:
                        xep_loai = "Cần cải thiện"
                        chien_luoc = "Nâng cao chất lượng dịch vụ, khảo sát feedback, chương trình khuyến mãi"
                    else:
                        xep_loai = "Cần quan tâm"
                        chien_luoc = "Tăng engagement, cải thiện trải nghiệm, giá cả hợp lý"
                    
                    strategy_data.append({
                        'Nhóm': cluster_id,
                        'Loại': info['type'],
                        'Xếp loại': xep_loai,
                        'Số lượng': f"{info['count']} khách hàng",
                        'Chi tiêu TB': f"${info['avg_amount']}",
                        'Đánh giá TB': f"{info['avg_rating']}/5.0",
                        'Chiến lược đề xuất': chien_luoc
                    })
                
                strategy_df = pd.DataFrame(strategy_data)
                st.dataframe(strategy_df, use_container_width=True)
                
        else:
            st.error("Dữ liệu không có cột 'Purchase Amount (USD)' hoặc 'Review Rating'")

    elif algo == "Naive Bayes":
        st.subheader("Dự đoán Rating theo Loại Sản phẩm - Naive Bayes")
        st.write("**Mục tiêu:** Dự đoán rating dựa trên loại sản phẩm và thông tin khách hàng")
        
        if 'Review Rating' in data_df.columns and 'Item Purchased' in data_df.columns:
            # Chuẩn bị dữ liệu cho Naive Bayes
            nb_data = data_df.copy()
            # Loại bỏ missing values trong Review Rating
            nb_data = nb_data.dropna(subset=['Review Rating'])
            
            feature_cols = ['Item Purchased']
            if 'Purchase Amount (USD)' in nb_data.columns:
                feature_cols.append('Purchase Amount (USD)')
            if 'Payment Method' in nb_data.columns:
                feature_cols.append('Payment Method')
            
            # Chuyển Review Rating thành categorical target (1-5 thành categories)
            nb_data['Rating_Category'] = pd.cut(nb_data['Review Rating'], 
                                               bins=[0, 2, 3, 4, 5], 
                                               labels=['Poor', 'Fair', 'Good', 'Excellent'])
            
            target = 'Rating_Category'
            
            # Hiển thị dữ liệu đầu vào
            st.write("**Dữ liệu đầu vào (Naive Bayes):**")
            nb_input = nb_data[feature_cols + [target]].copy()
            st.dataframe(nb_input.head(20))
            
            test_size = st.slider("Tỉ lệ test", 0.1, 0.5, 0.2)
            
            if st.button("Chạy Naive Bayes"):
                metrics, y_pred = run_nb(nb_data, target=target, feature_columns=feature_cols, test_size=test_size)
                
                st.subheader("Kết quả dự đoán Rating:")
                st.write(f"**Accuracy:** {metrics['accuracy']:.2%}")
                st.write(f"**Precision:** {metrics['precision']:.2%}")
                st.write(f"**Recall:** {metrics['recall']:.2%}")
                st.write(f"**F1-score:** {metrics['f1_score']:.2%}")
                
                # Ma trận nhầm lẫn
                if 'confusion_matrix' in metrics:
                    st.subheader("Ma trận nhầm lẫn")
                    cm_df = pd.DataFrame(metrics['confusion_matrix'], 
                                       index=[f'Thực tế {i}' for i in range(len(metrics['confusion_matrix']))],
                                       columns=[f'Dự đoán {i}' for i in range(len(metrics['confusion_matrix'][0]))])
                    st.dataframe(cm_df)
                
                # Diễn giải kinh doanh
                st.subheader("Diễn giải kinh doanh:")
                st.write("• **Mục đích:** Dự đoán rating để cải thiện chất lượng sản phẩm")
                st.write("• **Ứng dụng:** Tập trung vào sản phẩm có rating thấp để cải thiện")
                st.write("• **Chiến lược:** Ưu tiên phát triển sản phẩm trong category có rating cao")
        else:
            st.error("Dữ liệu không có cột 'Review Rating' hoặc 'Item Purchased'")

    elif algo == "Decision-Tree-CART":
        st.subheader("Kế hoạch Inventory - Decision Tree CART")
        st.write("**Mục tiêu:** Quyết định nhập hàng dựa trên lịch sử bán và mùa vụ")
        
        if 'Purchase Amount (USD)' in data_df.columns and 'Item Purchased' in data_df.columns:
            # Tạo target cho inventory decision
            cart_data = data_df.copy()
            if 'Date Purchase' in cart_data.columns:
                cart_data['Date Purchase'] = pd.to_datetime(cart_data['Date Purchase'], format='%d-%m-%Y')
                cart_data['Month'] = cart_data['Date Purchase'].dt.month
            
            # Tạo target: 1 nếu nên nhập hàng (doanh thu cao), 0 nếu không
            inventory_data = cart_data.groupby('Item Purchased').agg({
                'Purchase Amount (USD)': 'mean',
                'Month': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
            }).reset_index()
            
            # Tạo binary target cho inventory decision
            median_amount = inventory_data['Purchase Amount (USD)'].median()
            inventory_data['Should_Restock'] = (inventory_data['Purchase Amount (USD)'] > median_amount).astype(int)
            
            feature_cols = ['Purchase Amount (USD)', 'Month']
            target = 'Should_Restock'
            
            # Hiển thị dữ liệu đầu vào
            st.write("**Dữ liệu đầu vào (Decision Tree CART):**")
            cart_input = inventory_data[feature_cols + [target]].copy()
            st.dataframe(cart_input.head(20))
            
            max_depth = st.slider("Max depth", 1, 10, 5)
            min_split = st.slider("Min samples split", 2, 10, 2)
            
            if st.button("Chạy CART"):
                metrics = run_dt_cart(inventory_data, target=target, feature_columns=feature_cols, max_depth=max_depth, min_samples_split=min_split)
                
                st.subheader("Kết quả quyết định nhập hàng:")
                st.write(f"**Accuracy:** {metrics['accuracy']:.2%}")
                
                # Diễn giải kinh doanh
                st.subheader("Diễn giải kinh doanh:")
                st.write("• **Mục đích:** Quyết định nhập hàng dựa trên lịch sử bán")
                st.write("• **Ứng dụng:** Tối ưu hóa inventory, tránh tồn kho")
                st.write("• **Chiến lược:** Tập trung nhập hàng vào category có doanh thu cao")
        else:
            st.error("Dữ liệu không có cột 'Purchase Amount (USD)' hoặc 'Item Purchased'")

    elif algo == "Decision-Tree-ID3":
        st.subheader("Quality Control - Decision Tree ID3")
        st.write("**Mục tiêu:** Phân loại sản phẩm có vấn đề dựa trên rating và feedback")
        
        if 'Review Rating' in data_df.columns and 'Item Purchased' in data_df.columns:
            # Tạo target cho quality control: 1 nếu có vấn đề (rating thấp), 0 nếu không
            quality_data = data_df.copy()
            # Loại bỏ missing values trong Review Rating
            quality_data = quality_data.dropna(subset=['Review Rating'])
            
            # Tạo binary target: 1 nếu rating <= 3 (có vấn đề), 0 nếu rating > 3 (tốt)
            quality_data['Has_Quality_Issue'] = (quality_data['Review Rating'] <= 3).astype(int)
            
            feature_cols = ['Item Purchased', 'Purchase Amount (USD)']
            if 'Payment Method' in quality_data.columns:
                feature_cols.append('Payment Method')
            
            target = 'Has_Quality_Issue'
            
            # Hiển thị dữ liệu đầu vào
            st.write("**Dữ liệu đầu vào (Decision Tree ID3):**")
            id3_input = quality_data[feature_cols + [target]].copy()
            st.dataframe(id3_input.head(20))
            
            max_depth = st.slider("Max depth", 1, 10, 5)
            min_split = st.slider("Min samples split", 2, 10, 2)
            
            if st.button("Chạy ID3"):
                metrics = run_dt_id3(quality_data, target=target, feature_columns=feature_cols, max_depth=max_depth, min_samples_split=min_split)
                
                st.subheader("Kết quả kiểm soát chất lượng:")
                st.write(f"**Accuracy:** {metrics['accuracy']:.2%}")
                st.write(f"**Precision:** {metrics['precision']:.2%}")
                st.write(f"**Recall:** {metrics['recall']:.2%}")
                
                # Diễn giải kinh doanh
                st.subheader("Diễn giải kinh doanh:")
                st.write("• **Mục đích:** Phát hiện sản phẩm có vấn đề chất lượng")
                st.write("• **Ứng dụng:** Kiểm soát chất lượng, cải thiện sản phẩm")
                st.write("• **Chiến lược:** Tập trung vào sản phẩm có rating thấp để cải thiện")
        else:
            st.error("Dữ liệu không có cột 'Review Rating' hoặc 'Item Purchased'")

    elif algo == "Decision-Tree-Quinlan":
        st.subheader("Seasonal Planning - Decision Tree Quinlan (C4.5)")
        st.write("**Mục tiêu:** Dự đoán sản phẩm phù hợp theo quý")
        
        if 'Date Purchase' in data_df.columns and 'Item Purchased' in data_df.columns:
            # Chuẩn bị dữ liệu cho seasonal planning
            seasonal_data = data_df.copy()
            seasonal_data['Date Purchase'] = pd.to_datetime(seasonal_data['Date Purchase'], format='%d-%m-%Y')
            seasonal_data['Quarter'] = seasonal_data['Date Purchase'].dt.quarter
            
            # Tạo target: Quarter (1-4)
            feature_cols = ['Item Purchased', 'Purchase Amount (USD)']
            if 'Review Rating' in seasonal_data.columns:
                seasonal_data = seasonal_data.dropna(subset=['Review Rating'])
                feature_cols.append('Review Rating')
            if 'Payment Method' in seasonal_data.columns:
                feature_cols.append('Payment Method')
            
            target = 'Quarter'
            
            # Hiển thị dữ liệu đầu vào
            st.write("**Dữ liệu đầu vào (Decision Tree Quinlan):**")
            quinlan_input = seasonal_data[feature_cols + [target]].copy()
            st.dataframe(quinlan_input.head(20))
            
            max_depth = st.slider("Max depth", 1, 10, 5)
            min_split = st.slider("Min samples split", 2, 10, 2)
            
            if st.button("Chạy Quinlan"):
                metrics = run_dt_quinlan(seasonal_data, target=target, feature_columns=feature_cols, max_depth=max_depth, min_samples_split=min_split)
                
                st.subheader("Kết quả dự đoán mùa vụ:")
                st.write(f"**Accuracy:** {metrics['accuracy']:.2%}")
                
                # Diễn giải kinh doanh
                st.subheader("Diễn giải kinh doanh:")
                st.write("• **Mục đích:** Dự đoán sản phẩm phù hợp theo quý")
                st.write("• **Ứng dụng:** Kế hoạch sản phẩm theo mùa, marketing mùa vụ")
                st.write("• **Chiến lược:** Chuẩn bị inventory và marketing phù hợp với từng quý")
                
                # Hiển thị phân bố theo quý
                quarter_dist = seasonal_data['Quarter'].value_counts().sort_index()
                quarter_names = {1: 'Q1 (Winter)', 2: 'Q2 (Spring)', 3: 'Q3 (Summer)', 4: 'Q4 (Fall)'}
                st.subheader("Phân bố giao dịch theo quý:")
                for q, count in quarter_dist.items():
                    st.write(f"• **{quarter_names.get(q, f'Q{q}')}**: {count} giao dịch")
        else:
            st.error("Dữ liệu không có cột 'Date Purchase' hoặc 'Item Purchased'")

    elif algo == "Apriori":
        st.subheader("Thuật Toán Apriori (Tập phổ biến và luật liên kết)")
        st.write("**Mục tiêu:** Thuật toán Apriori được dùng trong bài toán để đưa ra các sản phẩm khác hàng thường được chọn kèm mỗi lần mua sắm, từ đó giúp đưa ra gợi ý bố trí sản phẩm hợp lý để tăng hiệu quả bán hàng.")
        
        # Hiển thị dữ liệu đầu vào cho Apriori
        if 'Customer Reference ID' in data_df.columns and 'Item Purchased' in data_df.columns:
            # Tạo transactions theo customer - mỗi customer có nhiều items
            transactions_df = data_df[['Customer Reference ID', 'Item Purchased']].copy()
            
            st.write("**Dữ liệu đầu vào:**")
            st.dataframe(transactions_df)
        else:
            st.error("Dữ liệu không có cột 'Customer Reference ID' hoặc 'Item Purchased'")
        
        min_sup = st.slider("Min support", 0.01, 0.2, 0.05, help="tần số trong bao nhiêu phần trăm dữ liệu thì những điều ở vế trái và vế phải cùng xảy ra")
        min_conf = st.slider("Min confidence", 0.2, 0.8, 0.4, help="độ mạnh nếu vế trái xảy ra thì có bao nhiêu khả năng vế phải xảy ra")
        top_k = st.slider("Số luật tốt nhất (top-k)", 3, 15, 5, help="Số luật tốt nhất hiển thị")

        if st.button("Chạy Apriori"):
            # Chuẩn bị dữ liệu cho Apriori: gom theo Customer ID
            if 'Customer Reference ID' in data_df.columns and 'Item Purchased' in data_df.columns:
                st.info(f"**Đang phân tích:** {len(transactions_df)} giao dịch từ {transactions_df['Customer Reference ID'].nunique()} khách hàng")
                
                # Tạo transactions list
                transactions_list = transactions_df.groupby('Customer Reference ID')['Item Purchased'].apply(list).tolist()
                transactions_list = [items for items in transactions_list if items]  # Loại bỏ empty
                
                # Tạo One-Hot Matrix để hiển thị
                st.write("**Ma trận biểu diễn tập giao dịch:**")
                try:
                    from mlxtend.preprocessing import TransactionEncoder
                    
                    if transactions_list:
                        te = TransactionEncoder()
                        arr = te.fit_transform(transactions_list)
                        ohe_df = pd.DataFrame(arr, columns=te.columns_)
                        ohe_df.index = [f"Customer_{i+1}" for i in range(len(ohe_df))]
                        
                        # Chỉ hiển thị một phần để không quá dài
                        display_cols = min(10, len(ohe_df.columns))
                        st.dataframe(ohe_df.iloc[:15, :display_cols])
                        
                        if len(ohe_df.columns) > display_cols:
                            st.info(f"Hiển thị {display_cols}/{len(ohe_df.columns)} cột. Tổng cộng có {len(ohe_df)} transactions và {len(ohe_df.columns)} sản phẩm.")
                    else:
                        st.warning("Không có dữ liệu để tạo ma trận")
                except Exception as e:
                    st.error(f"Lỗi tạo One-Hot Matrix: {str(e)}")
                
                # Chạy thuật toán Apriori               
                frequent, rules = run_apriori(transactions_df, min_support=min_sup, min_confidence=min_conf, top_k=top_k)
                
                # Hiển thị kết quả
                st.write("**Tập phổ biến thỏa min-support:**", min_sup)
                if not frequent.empty:
                    st.dataframe(frequent.head(20))
                    st.write(f"Tìm thấy {len(frequent)} tập phổ biến")
                else:
                    st.warning("Không tìm thấy itemsets phổ biến. Hãy giảm min_support.")
                
                st.write("**Luật liên kết sản phẩm:**")
                if not rules.empty:
                    st.dataframe(rules)
                    
                    # Phân tích kết quả
                    st.write("**Phân Tích Kết Quả:**")
                    for i, row in rules.iterrows():
                        st.write(f"• Nếu khách mua **{row['antecedents']}** → {row['confidence']:.1%} khả năng mua **{row['consequents']}** (lift={row['lift']:.1f})")
                        st.write(f"  → *Gợi ý:* Bố trí 2 sản phẩm này gần nhau trên kệ")
                else:
                    st.warning("Không tìm thấy luật liên kết nào. Hãy thử giảm min_support hoặc min_confidence.")
            else:
                st.error("Dữ liệu không có cột 'Customer Reference ID' hoặc 'Item Purchased'")

    elif algo == "Rough Set":
        st.subheader("Phân Tích Yếu Tố Ảnh Hưởng Đánh Giá - Rough Set")
        st.write("**Mục tiêu:** Tìm những yếu tố cốt lõi ảnh hưởng đến việc khách hàng để lại đánh giá tốt (Review Rating cao)")
        
        if 'Review Rating' in data_df.columns:
            # Hiển thị dữ liệu đầu vào
            from src.roughset import analyze_review_factors
            
            # Chuẩn bị dữ liệu để hiển thị
            analysis_data = data_df.dropna(subset=['Review Rating']).copy()
            analysis_data['High_Rating'] = (analysis_data['Review Rating'] >= 4.0).astype(int)
            
            # Xử lý missing values cho các cột hiển thị
            if 'Item Purchased' in analysis_data.columns:
                analysis_data['Item Purchased'] = analysis_data['Item Purchased'].fillna('Unknown')
            if 'Payment Method' in analysis_data.columns:
                analysis_data['Payment Method'] = analysis_data['Payment Method'].fillna('Unknown')
            if 'Purchase Amount (USD)' in analysis_data.columns:
                if analysis_data['Purchase Amount (USD)'].notna().any():
                    median_val = analysis_data['Purchase Amount (USD)'].median()
                    analysis_data['Purchase Amount (USD)'] = analysis_data['Purchase Amount (USD)'].fillna(median_val)
                else:
                    analysis_data['Purchase Amount (USD)'] = analysis_data['Purchase Amount (USD)'].fillna(0)
            
            # Chọn các cột cốt lõi để hiển thị
            display_cols = []
            if 'Item Purchased' in analysis_data.columns:
                display_cols.append('Item Purchased')
            if 'Payment Method' in analysis_data.columns:
                display_cols.append('Payment Method')
            if 'Purchase Amount (USD)' in analysis_data.columns:
                display_cols.append('Purchase Amount (USD)')
            display_cols.extend(['Review Rating', 'High_Rating'])
            
            st.write("**Dữ liệu đầu vào:**")
            st.dataframe(analysis_data[display_cols].head(20))
            
            if st.button("Chạy Rough Set"):
                # Phân tích các yếu tố
                insights = analyze_review_factors(data_df)
                
                st.subheader("Các Yếu Tố Cốt Lõi Ảnh Hưởng Đánh Giá:")
                for i, factor in enumerate(insights['important_factors'], 1):
                    st.write(f"**{i}. {factor}**")
                
                # Thống kê tổng quan
                st.subheader("Thống Kê Tổng Quan:")
                summary = insights['summary']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tổng số đánh giá", summary['total_reviews'])
                with col2:
                    st.metric("Đánh giá tốt", f"{summary['high_rating_count']} ({summary['high_rating_rate']:.1%})")
                with col3:
                    st.metric("Đánh giá trung bình", f"{summary['avg_rating']}/5.0")
                
                # Phân tích chi tiết từng yếu tố
                st.subheader("Phân Tích Chi Tiết Các Yếu Tố:")
                for i, factor in enumerate(insights['important_factors'], 1):
                    st.write(f"**{i}. {factor}**")
                    
                    factor_info = insights['factor_analysis'][factor]
                    
                    if factor_info['type'] == 'categorical':
                        # Yếu tố categorical
                        st.write(f"**Loại tốt nhất:** {factor_info['best_category']} ({factor_info['best_rate']:.1%} đánh giá tốt)")
                        st.write("**Chi tiết theo từng loại:**")
                        
                        details = factor_info['details']
                        if details:  # Kiểm tra nếu có dữ liệu
                            for category, stats in details.items():
                                if isinstance(stats, dict) and 'Total_Reviews' in stats:
                                    total = stats['Total_Reviews']
                                    rate = stats['High_Rating_Rate']
                                    st.write(f"• **{category}**: {total} đánh giá, {rate:.1%} tốt")
                                else:
                                    st.write(f"• **{category}**: Dữ liệu không hợp lệ")
                        else:
                            st.write("• Không có dữ liệu chi tiết")
                            
                    else:
                        # Yếu tố numerical
                        st.write("**So sánh giá trị trung bình:**")
                        st.write(f"• Đánh giá tốt (≥4.0): **{factor_info['high_rating_avg']}**")
                        st.write(f"• Đánh giá thấp (<4.0): **{factor_info['low_rating_avg']}**")
                        st.write(f"• Chênh lệch: **{factor_info['difference']}** ({factor_info['impact']})")
                    
                    st.write("---")
                
                # Khuyến nghị cụ thể
                st.subheader("Khuyến Nghị Cụ Thể:")
                for recommendation in insights['recommendations']:
                    st.write(f"• {recommendation}")
        else:
            st.error("Dữ liệu không có cột 'Review Rating'")

    else:
        st.info("Chọn thuật toán ở sidebar để chạy.")


if __name__ == "__main__":
    main()


