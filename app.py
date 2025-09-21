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
        return load_csv("data/datasets/Fashion_Retail_Sales.csv")
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
        st.subheader("Thuật toán phân cụm K-means")
        st.write("**Bài toán:** Phân nhóm khách hàng theo mức chi tiêu và độ hài lòng")
        st.write("**Mục tiêu:** Phân nhóm khách hàng dựa trên số tiền chi tiêu và mức độ hài lòng, giúp doanh nghiệp nhận diện các phân khúc khách hàng tiềm năng để tối ưu hóa chiến lược marketing và chăm sóc khách hàng.")
        # Chuẩn bị dữ liệu cho K-means phân nhóm khách hàng
        if 'Purchase Amount (USD)' in data_df.columns and 'Review Rating' in data_df.columns:
            # Xử lý missing values
            kmeans_data = data_df.dropna(subset=['Purchase Amount (USD)', 'Review Rating']).copy()
            
            st.write("**Dữ liệu đầu vào:**")
            kmeans_input = kmeans_data[['Customer Reference ID', 'Purchase Amount (USD)', 'Review Rating']].copy()
            st.dataframe(kmeans_input)
        else:
            st.error("Dữ liệu không có cột 'Purchase Amount (USD)' hoặc 'Review Rating'")

    # Phần chạy thuật toán K-means
    if algo == "K-means":
        
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
        st.subheader("Thuật toán Naive Bayes")
        st.write("**Bài toán:** Dự đoán khách hàng có khả năng quay lại mua hàng hay không")
        st.write("**Mục tiêu:** Xây dựng mô hình dự đoán khách hàng có khả năng quay lại mua hàng hay không, giúp doanh nghiệp tối ưu hóa chiến lược marketing và chăm sóc khách hàng.")
        
        if 'Review Rating' in data_df.columns and 'Item Purchased' in data_df.columns:
            # Chuẩn bị dữ liệu cho Naive Bayes
            nb_data = data_df.copy()
            # Loại bỏ missing values
            nb_data = nb_data.dropna(subset=['Review Rating', 'Purchase Amount (USD)'])
            
            # Hiển thị dữ liệu đầu vào
            st.write("**Dữ liệu đầu vào:**")
            if 'Will_Return' in nb_data.columns:
                # Sử dụng cột có sẵn
                nb_input = nb_data[['Review Rating', 'Purchase Amount (USD)', 'Item Purchased', 'Payment Method', 'Will_Return']].copy()
                st.dataframe(nb_input)
            else:
                # Tạo target mới (fallback)
                avg_amount = nb_data['Purchase Amount (USD)'].mean()
                nb_data['Will_Return'] = (
                    (nb_data['Review Rating'] >= 4.0) & 
                    (nb_data['Purchase Amount (USD)'] >= avg_amount * 0.8)
                ).astype(int)
                
                nb_input = nb_data[['Review Rating', 'Purchase Amount (USD)', 'Item Purchased', 'Payment Method', 'Will_Return']].copy()
                st.dataframe(nb_input.head(20))
                
                # Hiển thị thông tin về target
                will_return_count = nb_data['Will_Return'].sum()
                total_count = len(nb_data)
                st.info(f"**Ngưỡng dự đoán:** Rating ≥ 4.0 và chi tiêu ≥ ${avg_amount*0.8:.0f} ({will_return_count}/{total_count} khách hàng)")
            
            if st.button("Chạy Naive Bayes"):
                metrics, y_pred = run_nb(nb_data, test_size=0.2)
                
                st.success(f"✅ Hoàn thành dự đoán")
                
                # Hiển thị kết quả dự đoán đơn giản
                st.subheader("Kết Quả Dự Đoán")
                
                prob_table = metrics['probability_table']
                
                # Hiển thị xác suất tổng quan
                col1, col2 = st.columns(2)
                with col1:
                    will_return_prob = prob_table['prior_probabilities'].get(1, 0)
                    st.metric("Khách hàng sẽ mua lại", f"{will_return_prob:.1%}")
                
                with col2:
                    no_return_prob = prob_table['prior_probabilities'].get(0, 0)
                    st.metric("Khách hàng không mua lại", f"{no_return_prob:.1%}")
                
                # Phân tích chi tiết theo nhóm
                st.subheader("Phân Tích Chi Tiết Theo Nhóm")
                
                from src.naive_bayes import create_customer_insights, generate_business_recommendations
                insights = create_customer_insights(nb_data, prob_table)
                
                # Phân tích theo sản phẩm
                st.write("**Phân tích theo sản phẩm:**")
                product_df = insights['product_analysis'].reset_index()
                product_df.columns = ['Sản phẩm', 'Tổng số', 'Sẽ mua lại', 'Tỷ lệ mua lại']
                product_df['Tỷ lệ mua lại'] = product_df['Tỷ lệ mua lại'].apply(lambda x: f"{x:.1%}")
                st.dataframe(product_df, use_container_width=True)
                
                # Phân tích theo phương thức thanh toán
                st.write("**Phân tích theo phương thức thanh toán:**")
                payment_df = insights['payment_analysis'].reset_index()
                payment_df.columns = ['Phương thức', 'Tổng số', 'Sẽ mua lại', 'Tỷ lệ mua lại']
                payment_df['Tỷ lệ mua lại'] = payment_df['Tỷ lệ mua lại'].apply(lambda x: f"{x:.1%}")
                st.dataframe(payment_df, use_container_width=True)
                
                # Phân tích theo mức chi tiêu
                st.write("**Phân tích theo mức chi tiêu:**")
                spending_df = insights['spending_analysis'].reset_index()
                spending_df.columns = ['Mức chi tiêu', 'Tổng số', 'Sẽ mua lại', 'Tỷ lệ mua lại']
                spending_df['Tỷ lệ mua lại'] = spending_df['Tỷ lệ mua lại'].apply(lambda x: f"{x:.1%}")
                st.dataframe(spending_df, use_container_width=True)
                
                # Khuyến nghị kinh doanh
                st.subheader("Tổng Kết Đề Xuất")
                recommendations = generate_business_recommendations(insights)
                
                for rec in recommendations:
                    st.write(rec)
        else:
            st.error("Dữ liệu không có cột 'Review Rating' hoặc 'Item Purchased'")

    elif algo == "Decision-Tree-CART":
        st.subheader("Thuật toán cây quyết định - Decision Tree CART")
        st.write("**Bài toán:** Quyết định sản phẩm nên nhập/nên dừng")
        st.write("**Thuật toán CART:** Sử dụng Gini để xây dựng cây quyết định")
        st.write("**Yếu tố ảnh hưởng:** Sales_Volume, Profit_Margin, Customer_Demand, Seasonality")
        st.write("**Mục tiêu:** Restock (Yes/No) - dựa trên Sales cao, Rating tốt, Giá trị hợp lý")
        
        if 'Review Rating' in data_df.columns and 'Item Purchased' in data_df.columns and 'Purchase Amount (USD)' in data_df.columns:
            # Hiển thị dữ liệu đầu vào
            st.write("**Dữ liệu đầu vào:**")
            st.dataframe(data_df[['Item Purchased', 'Purchase Amount (USD)', 'Payment Method', 'Review Rating', 'Date Purchase']].head(20))
            
            if st.button("Chạy CART"):
                metrics = run_dt_cart(data_df)
                
                st.success("✅ Hoàn thành phân tích Decision Tree CART")
                
                # Hiển thị thống kê tổng quan
                data_summary = metrics['data_summary']
                st.subheader("1. Thống Kê Tổng Quan")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tổng mẫu", data_summary['total_samples'])
                with col2:
                    st.metric("Nên nhập", f"{data_summary['should_restock_count']} ({data_summary['should_restock_count']/data_summary['total_samples']:.1%})")
                with col3:
                    st.metric("Nên dừng", f"{data_summary['should_stop_count']} ({data_summary['should_stop_count']/data_summary['total_samples']:.1%})")
                with col4:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                
                # Hiển thị bảng Gini Impurity và Information Gain
                gini_info = metrics['gini_info']
                st.subheader("2. Bảng Gini Impurity và Information Gain")
                gain_data = []
                for feature, gain_info in gini_info['feature_gains'].items():
                    gain_data.append({
                        'Thuộc tính': feature,
                        'Information Gain': f"{gain_info['information_gain']:.3f}",
                        'Weighted Gini': f"{gain_info['weighted_gini']:.3f}"
                    })
                
                gain_df = pd.DataFrame(gain_data)
                st.dataframe(gain_df, use_container_width=True)
                
                # Hiển thị chi tiết Gini
                st.subheader("3. Chi Tiết Gini Impurity Theo Thuộc Tính")
                
                for feature, gain_info in gini_info['feature_gains'].items():
                    with st.expander(f"📊 {feature} (Gain: {gain_info['information_gain']:.3f})"):
                        st.write("**Phân bố thuộc tính:**")
                        feature_dist = gain_info['feature_distribution']
                        for value, count in feature_dist.items():
                            st.write(f"• {value}: {count} mẫu")
                        
                        st.write("**Chi tiết Gini Impurity:**")
                        gini_details = gain_info['gini_details']
                        for value, details in gini_details.items():
                            st.write(f"**{value}:**")
                            st.write(f"  - Số mẫu: {details['count']}")
                            st.write(f"  - Gini: {details['gini']:.3f}")
                            st.write(f"  - Phân bố: {details['target_distribution']}")
                
                # Hiển thị sơ đồ cây quyết định trực quan
                st.subheader("4. Sơ Đồ Cây Quyết Định")
                
                tree_figure = metrics.get('decision_tree_figure')
                if tree_figure is not None:
                    st.write("**Sơ đồ cây quyết định trực quan:**")
                    st.pyplot(tree_figure)
                else:
                    st.write("**Sơ đồ trực quan không khả dụng, hiển thị dạng text:**")
                    tree_rules = metrics['decision_tree_rules']
                    st.text_area("Cây quyết định CART:", tree_rules, height=200)
                
                # Phân tích kết quả
                st.subheader("5. Phân Tích Kết Quả")
                restock_analysis = metrics['restock_analysis']
                
                # Phân tích theo Sales_Volume
                st.write("**Phân tích theo khối lượng bán:**")
                sales_data = []
                for sales_volume, stats in restock_analysis['Sales_Volume'].items():
                    sales_data.append({
                        'Khối lượng bán': sales_volume,
                        'Số mẫu': stats['count'],
                        'Nên nhập': stats['should_restock'],
                        'Tỷ lệ nên nhập': f"{stats['restock_rate']}%"
                    })
                
                sales_df = pd.DataFrame(sales_data)
                st.dataframe(sales_df, use_container_width=True)
                
                # Phân tích theo Profit_Margin
                st.write("**Phân tích theo tỷ suất lợi nhuận:**")
                profit_data = []
                for profit_margin, stats in restock_analysis['Profit_Margin'].items():
                    profit_data.append({
                        'Tỷ suất lợi nhuận': profit_margin,
                        'Số mẫu': stats['count'],
                        'Nên nhập': stats['should_restock'],
                        'Tỷ lệ nên nhập': f"{stats['restock_rate']}%"
                    })
                
                profit_df = pd.DataFrame(profit_data)
                st.dataframe(profit_df, use_container_width=True)
                
                # Phân tích theo Customer_Demand
                st.write("**Phân tích theo mức độ quan tâm khách hàng:**")
                demand_data = []
                for customer_demand, stats in restock_analysis['Customer_Demand'].items():
                    demand_data.append({
                        'Mức độ quan tâm': customer_demand,
                        'Số mẫu': stats['count'],
                        'Nên nhập': stats['should_restock'],
                        'Tỷ lệ nên nhập': f"{stats['restock_rate']}%"
                    })
                
                demand_df = pd.DataFrame(demand_data)
                st.dataframe(demand_df, use_container_width=True)
                
                # Phân tích theo Seasonality
                st.write("**Phân tích theo tính mùa vụ:**")
                season_data = []
                for seasonality, stats in restock_analysis['Seasonality'].items():
                    season_data.append({
                        'Mùa vụ': seasonality,
                        'Số mẫu': stats['count'],
                        'Nên nhập': stats['should_restock'],
                        'Tỷ lệ nên nhập': f"{stats['restock_rate']}%"
                    })
                
                season_df = pd.DataFrame(season_data)
                st.dataframe(season_df, use_container_width=True)
                
                # Kết luận
                st.subheader("6. Kết Luận")
                best_feature = list(gini_info['feature_gains'].keys())[0]
                best_gain = gini_info['feature_gains'][best_feature]['information_gain']
                
                st.write(f"**Thuộc tính quan trọng nhất**: {best_feature} (Gain: {best_gain:.3f})")
                overview = restock_analysis['overview']
                if overview['restock_rate'] > 50:
                    st.write("Tỷ lệ sản phẩm nên nhập cao - tập trung mở rộng inventory")
                else:
                    st.write("Tỷ lệ sản phẩm nên nhập thấp - cần cải thiện chất lượng sản phẩm")
                
        else:
            st.error("Dữ liệu không có đủ cột cần thiết: 'Review Rating', 'Item Purchased', 'Purchase Amount (USD)'")

    elif algo == "Decision-Tree-ID3":
        st.subheader("Thuật toán cây quyết định - Decision Tree ID3")
        st.write("**Bài toán:** Dự đoán khách hàng có mua hàng hay không")
        st.write("**Mục tiêu:** Dự đoán khách hàng có mua hàng hay không, giúp doanh nghiệp tối ưu hóa chiến lược marketing và chăm sóc khách hàng.")
        st.write("**Yếu tố ảnh hưởng:** Item_Type, Price_Range, Payment_Preference, Customer_Type")
        st.write("**Mục tiêu:** Will_Buy (Yes/No) - dựa trên Review Rating ≥ 3.5 và giá trị ≥ $100")
        
        if 'Review Rating' in data_df.columns and 'Item Purchased' in data_df.columns and 'Purchase Amount (USD)' in data_df.columns:
            # Hiển thị dữ liệu đầu vào
            st.write("**Dữ liệu đầu vào:**")
            st.dataframe(data_df[['Item Purchased', 'Purchase Amount (USD)', 'Payment Method', 'Review Rating', 'Will_Return']].head(20))
            
            if st.button("Chạy ID3"):
                metrics = run_dt_id3(data_df)
                
                st.success("✅ Hoàn thành phân tích Decision Tree ID3")
                
                # Hiển thị thống kê tổng quan
                data_summary = metrics['data_summary']
                st.subheader("1. Thống Kê Tổng Quan")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tổng mẫu", data_summary['total_samples'])
                with col2:
                    st.metric("Sẽ mua", f"{data_summary['will_buy_count']} ({data_summary['will_buy_count']/data_summary['total_samples']:.1%})")
                with col3:
                    st.metric("Không mua", f"{data_summary['will_not_buy_count']} ({data_summary['will_not_buy_count']/data_summary['total_samples']:.1%})")
                with col4:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                
                # Hiển thị bảng Information Gain
                entropy_info = metrics['entropy_info']
                st.subheader("2. Bảng Information Gain")
                gain_data = []
                for feature, gain_info in entropy_info['feature_gains'].items():
                    gain_data.append({
                        'Thuộc tính': feature,
                        'Information Gain': f"{gain_info['information_gain']:.3f}",
                        'Weighted Entropy': f"{gain_info['weighted_entropy']:.3f}"
                    })
                
                gain_df = pd.DataFrame(gain_data)
                st.dataframe(gain_df, use_container_width=True)
                
                # Hiển thị chi tiết Entropy
                st.subheader("3. Chi Tiết Entropy Theo Thuộc Tính")
                
                for feature, gain_info in entropy_info['feature_gains'].items():
                    with st.expander(f"📊 {feature} (Gain: {gain_info['information_gain']:.3f})"):
                        st.write("**Phân bố thuộc tính:**")
                        feature_dist = gain_info['feature_distribution']
                        for value, count in feature_dist.items():
                            st.write(f"• {value}: {count} mẫu")
                        
                        st.write("**Chi tiết Entropy:**")
                        entropy_details = gain_info['entropy_details']
                        for value, details in entropy_details.items():
                            st.write(f"**{value}:**")
                            st.write(f"  - Số mẫu: {details['count']}")
                            st.write(f"  - Entropy: {details['entropy']:.3f}")
                            st.write(f"  - Phân bố: {details['target_distribution']}")
                
                # Hiển thị sơ đồ cây quyết định trực quan
                st.subheader("4. Sơ Đồ Cây Quyết Định")
                
                tree_figure = metrics.get('decision_tree_figure')
                if tree_figure is not None:
                    st.write("**Sơ đồ cây quyết định trực quan:**")
                    st.pyplot(tree_figure)
                else:
                    st.write("**Sơ đồ trực quan không khả dụng, hiển thị dạng text:**")
                    tree_rules = metrics['decision_tree_rules']
                    st.text_area("Cây quyết định ID3:", tree_rules, height=200)
                
                # Phân tích kết quả
                st.subheader("5. Phân Tích Kết Quả")
                purchase_analysis = metrics['purchase_analysis']
                
                # Phân tích theo Item_Type
                st.write("**Phân tích theo loại sản phẩm:**")
                item_type_data = []
                for item_type, stats in purchase_analysis['Item_Type'].items():
                    item_type_data.append({
                        'Loại sản phẩm': item_type,
                        'Số mẫu': stats['count'],
                        'Sẽ mua': stats['will_buy'],
                        'Tỷ lệ mua': f"{stats['buy_rate']}%"
                    })
                
                item_type_df = pd.DataFrame(item_type_data)
                st.dataframe(item_type_df, use_container_width=True)
                
                # Phân tích theo Price_Range
                st.write("**Phân tích theo mức giá:**")
                price_data = []
                for price_range, stats in purchase_analysis['Price_Range'].items():
                    price_data.append({
                        'Mức giá': price_range,
                        'Số mẫu': stats['count'],
                        'Sẽ mua': stats['will_buy'],
                        'Tỷ lệ mua': f"{stats['buy_rate']}%"
                    })
                
                price_df = pd.DataFrame(price_data)
                st.dataframe(price_df, use_container_width=True)
                
                # Phân tích theo Customer_Type
                st.write("**Phân tích theo loại khách hàng:**")
                customer_data = []
                for customer_type, stats in purchase_analysis['Customer_Type'].items():
                    customer_data.append({
                        'Loại khách hàng': customer_type,
                        'Số mẫu': stats['count'],
                        'Sẽ mua': stats['will_buy'],
                        'Tỷ lệ mua': f"{stats['buy_rate']}%"
                    })
                
                customer_df = pd.DataFrame(customer_data)
                st.dataframe(customer_df, use_container_width=True)
                
                # Kết luận
                st.subheader("6. Kết Luận")
                best_feature = list(entropy_info['feature_gains'].keys())[0]
                best_gain = entropy_info['feature_gains'][best_feature]['information_gain']
                
                st.write(f"**Thuộc tính quan trọng nhất**: {best_feature} (Gain: {best_gain:.3f})")
                st.write("**Mục đích**: Dự đoán khả năng mua hàng của khách hàng")
                st.write("**Ứng dụng**: Tối ưu hóa marketing, cá nhân hóa trải nghiệm khách hàng")
                
                # Khuyến nghị
                st.write("**Khuyến nghị:**")
                overview = purchase_analysis['overview']
                if overview['buy_rate'] > 50:
                    st.write("• Tỷ lệ mua hàng cao - tập trung duy trì chất lượng dịch vụ")
                else:
                    st.write("• Tỷ lệ mua hàng thấp - cần cải thiện trải nghiệm khách hàng")
                
                st.write(f"• Tổng cộng {overview['total_samples']} mẫu, {overview['will_buy']} khách hàng có khả năng mua ({overview['buy_rate']}%)")
        else:
            st.error("Dữ liệu không có đủ cột cần thiết: 'Review Rating', 'Item Purchased', 'Purchase Amount (USD)'")

    elif algo == "Decision-Tree-Quinlan":
        st.subheader("Thuật toán cây quyết định - Decision Tree Quinlan (C4.5)")
        st.write("**Bài toán:** Dự đoán xu hướng mùa vụ")
        st.write("**Mục tiêu:** Dự đoán xu hướng mùa vụ, giúp doanh nghiệp tối ưu hóa chiến lược sản phẩm và marketing mùa vụ.")
        st.write("**Thuật toán C4.5:** Sử dụng Gain Ratio để xây dựng cây quyết định")
        st.write("**Yếu tố ảnh hưởng:** Product_Category, Price_Level, Customer_Segment, Time_Period")
        st.write("**Output:** Seasonal_Trend (High/Low) - dựa trên Rating tốt, Giá cao, Thời kỳ phù hợp")
        
        if 'Review Rating' in data_df.columns and 'Item Purchased' in data_df.columns and 'Purchase Amount (USD)' in data_df.columns:
            # Hiển thị dữ liệu đầu vào
            st.write("**Dữ liệu đầu vào:**")
            st.dataframe(data_df[['Item Purchased', 'Purchase Amount (USD)', 'Payment Method', 'Review Rating', 'Date Purchase', 'Will_Return']].head(20))
            
            if st.button("Chạy Quinlan"):
                metrics = run_dt_quinlan(data_df)
                
                st.success("✅ Hoàn thành phân tích Decision Tree Quinlan (C4.5)")
                
                # Hiển thị thống kê tổng quan
                data_summary = metrics['data_summary']
                st.subheader("1. Thống Kê Tổng Quan")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tổng mẫu", data_summary['total_samples'])
                with col2:
                    st.metric("Xu hướng cao", f"{data_summary['high_trend_count']} ({data_summary['high_trend_count']/data_summary['total_samples']:.1%})")
                with col3:
                    st.metric("Xu hướng thấp", f"{data_summary['low_trend_count']} ({data_summary['low_trend_count']/data_summary['total_samples']:.1%})")
                with col4:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                
                # Hiển thị bảng Gain Ratio
                gain_ratio_info = metrics['gain_ratio_info']
                st.subheader("2. Bảng Entropy và Gain Ratio")
                gain_data = []
                for feature, gain_info in gain_ratio_info['feature_gains'].items():
                    gain_data.append({
                        'Thuộc tính': feature,
                        'Information Gain': f"{gain_info['information_gain']:.3f}",
                        'Split Information': f"{gain_info['split_information']:.3f}",
                        'Gain Ratio': f"{gain_info['gain_ratio']:.3f}"
                    })
                
                gain_df = pd.DataFrame(gain_data)
                st.dataframe(gain_df, use_container_width=True)
                
                # Hiển thị chi tiết Gain Ratio
                st.subheader("3. Chi Tiết Gain Ratio Theo Thuộc Tính")
                
                for feature, gain_info in gain_ratio_info['feature_gains'].items():
                    with st.expander(f"📊 {feature} (Gain Ratio: {gain_info['gain_ratio']:.3f})"):
                        st.write("**Phân bố thuộc tính:**")
                        feature_dist = gain_info['feature_distribution']
                        for value, count in feature_dist.items():
                            st.write(f"• {value}: {count} mẫu")
                        
                        st.write("**Chi tiết Entropy:**")
                        entropy_details = gain_info['entropy_details']
                        for value, details in entropy_details.items():
                            st.write(f"**{value}:**")
                            st.write(f"  - Số mẫu: {details['count']}")
                            st.write(f"  - Entropy: {details['entropy']:.3f}")
                            st.write(f"  - Phân bố: {details['target_distribution']}")
                
                # Hiển thị sơ đồ cây quyết định trực quan
                st.subheader("4. Sơ Đồ Cây Quyết Định")
                
                tree_figure = metrics.get('decision_tree_figure')
                if tree_figure is not None:
                    st.write("**Sơ đồ cây quyết định trực quan:**")
                    st.pyplot(tree_figure)
                else:
                    st.write("**Sơ đồ trực quan không khả dụng, hiển thị dạng text:**")
                    tree_rules = metrics['decision_tree_rules']
                    st.text_area("Cây quyết định C4.5:", tree_rules, height=200)
                
                # Phân tích kết quả
                st.subheader("5. Phân Tích Kết Quả")
                seasonal_analysis = metrics['seasonal_analysis']
                
                # Phân tích theo Product_Category
                st.write("**Phân tích theo danh mục sản phẩm:**")
                product_data = []
                for product_category, stats in seasonal_analysis['Product_Category'].items():
                    product_data.append({
                        'Danh mục sản phẩm': product_category,
                        'Số mẫu': stats['count'],
                        'Xu hướng cao': stats['high_trend'],
                        'Tỷ lệ xu hướng cao': f"{stats['high_trend_rate']}%"
                    })
                
                product_df = pd.DataFrame(product_data)
                st.dataframe(product_df, use_container_width=True)
                
                # Phân tích theo Price_Level
                st.write("**Phân tích theo mức giá:**")
                price_data = []
                for price_level, stats in seasonal_analysis['Price_Level'].items():
                    price_data.append({
                        'Mức giá': price_level,
                        'Số mẫu': stats['count'],
                        'Xu hướng cao': stats['high_trend'],
                        'Tỷ lệ xu hướng cao': f"{stats['high_trend_rate']}%"
                    })
                
                price_df = pd.DataFrame(price_data)
                st.dataframe(price_df, use_container_width=True)
                
                # Phân tích theo Customer_Segment
                st.write("**Phân tích theo phân khúc khách hàng:**")
                segment_data = []
                for customer_segment, stats in seasonal_analysis['Customer_Segment'].items():
                    segment_data.append({
                        'Phân khúc khách hàng': customer_segment,
                        'Số mẫu': stats['count'],
                        'Xu hướng cao': stats['high_trend'],
                        'Tỷ lệ xu hướng cao': f"{stats['high_trend_rate']}%"
                    })
                
                segment_df = pd.DataFrame(segment_data)
                st.dataframe(segment_df, use_container_width=True)
                
                # Phân tích theo Time_Period
                st.write("**Phân tích theo thời kỳ:**")
                time_data = []
                for time_period, stats in seasonal_analysis['Time_Period'].items():
                    time_data.append({
                        'Thời kỳ': time_period,
                        'Số mẫu': stats['count'],
                        'Xu hướng cao': stats['high_trend'],
                        'Tỷ lệ xu hướng cao': f"{stats['high_trend_rate']}%"
                    })
                
                time_df = pd.DataFrame(time_data)
                st.dataframe(time_df, use_container_width=True)
                
                # Kết luận
                st.subheader("6. Kết Luận")
                best_feature = list(gain_ratio_info['feature_gains'].keys())[0]
                best_gain_ratio = gain_ratio_info['feature_gains'][best_feature]['gain_ratio']
                
                st.write(f"**Thuộc tính quan trọng nhất**: {best_feature} (Gain Ratio: {best_gain_ratio:.3f})")
                st.write("**Mục đích**: Dự đoán xu hướng mùa vụ của sản phẩm")
                st.write("**Ứng dụng**: Kế hoạch sản phẩm theo mùa, marketing mùa vụ, tối ưu hóa inventory")
                
                # Khuyến nghị
                overview = seasonal_analysis['overview']
                if overview['high_trend_rate'] > 50:
                    st.write("Tỷ lệ xu hướng mùa vụ cao - tập trung phát triển sản phẩm theo mùa")
                else:
                    st.write("Tỷ lệ xu hướng mùa vụ thấp - cần cải thiện chiến lược mùa vụ")
                
        else:
            st.error("Dữ liệu không có đủ cột cần thiết: 'Review Rating', 'Item Purchased', 'Purchase Amount (USD)'")

    elif algo == "Apriori":
        st.subheader("Thuật Toán Apriori (Tập phổ biến và luật liên kết)")
        st.write("**Bài toán:** Đưa ra các sản phẩm khác hàng thường được chọn kèm mỗi lần mua sắm, từ đó giúp đưa ra gợi ý bố trí sản phẩm hợp lý để tăng hiệu quả bán hàng.")
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
        st.subheader("Thuật toán tập thô (Rough Set)")
        st.write("**Bài toán:** Phân Tích Yếu Tố Ảnh Hưởng Đánh Giá")
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
                
                # Đề xuất
                st.subheader("Đề xuất:")
                for recommendation in insights['recommendations']:
                    st.write(f"• {recommendation}")
        else:
            st.error("Dữ liệu không có cột 'Review Rating'")

    else:
        st.info("Chọn thuật toán ở sidebar để chạy.")


if __name__ == "__main__":
    main()


