import streamlit as st
import pandas as pd
from src.utils import load_csv
from src.kmeans import run_kmeans, plot_clusters, plot_elbow_method, plot_cluster_stats
from src.naive_bayes import run_nb
from src.dt_cart import run_dt_cart
from src.dt_id3 import run_dt_id3
from src.dt_c45 import run_dt_c45
from src.apriori import run_apriori
from src.roughset import reduce_attributes


st.set_page_config(page_title="Ứng Dụng Khai Thác Dữ Liệu Trong Lĩnh Vực Bán Hàng", layout="wide")


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
            "Rough Set",
            "Decision-Tree-ID3",
            "Decision-Tree-CART",
            "Decision-Tree-C4.5",
            "K-means",
            "Naive Bayes",
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
    if algo == "Apriori":
        if uploaded is not None:
            data_df = pd.read_csv(uploaded)
        else:
            try:
                data_df = pd.read_csv("data/datasets/transactions.csv")
            except Exception:
                data_df = pd.DataFrame()
        if data_df.empty:
            st.warning("Không tìm thấy dữ liệu transactions. Hãy tải CSV hoặc thêm vào data/datasets/transactions.csv")
            return
    else:
        if uploaded is not None:
            data_df = pd.read_csv(uploaded)
        else:
            data_df = load_default_customers()
        if data_df.empty:
            st.warning("Không tìm thấy dữ liệu customers. Hãy tải CSV hoặc thêm vào data/datasets/customers.csv")
            return

    st.subheader("Dữ liệu sử dụng")
    st.dataframe(data_df)

    if algo == "K-means":
        st.subheader("Tham số cho thuật toán K-means")
        numeric_cols = list(data_df.select_dtypes(include=["number"]).columns)
        feature_cols = st.multiselect("Chọn cột dùng làm đặc trưng", numeric_cols, default=[c for c in numeric_cols if c != "customer_id"])
        k = st.slider("Số cụm (k)", 2, 8, 3)
        max_iter = st.slider("Số vòng lặp tối đa", 100, 500, 300, step=50)

        if st.button("Chạy K-means"):
            labels, inertia = run_kmeans(data_df, feature_cols, k=k, max_iter=max_iter)
            st.success(f"Đã phân cụm: k={k}, inertia={inertia:.2f}")
            
            # Hiển thị kết quả phân cụm
            result = data_df.copy()
            result["cluster"] = labels
            st.dataframe(result[["customer_id", *feature_cols, "cluster"]])
            
            # Thống kê theo cụm
            st.subheader("Thống kê theo cụm")
            st.dataframe(result.groupby("cluster")[feature_cols].mean().round(2))
            
            # Biểu đồ trực quan hóa
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Biểu đồ phân cụm")
                if len(feature_cols) >= 2:
                    fig_clusters = plot_clusters(result, feature_cols, labels, k)
                    if fig_clusters:
                        st.pyplot(fig_clusters)
                else:
                    st.info("Cần ít nhất 2 đặc trưng để vẽ biểu đồ phân cụm")
            
            with col2:
                st.subheader("Elbow Method")
                fig_elbow = plot_elbow_method(data_df, feature_cols, max_k=8)
                st.pyplot(fig_elbow)
            
            # Biểu đồ thống kê
            st.subheader("Biểu đồ thống kê theo cụm")
            fig_stats = plot_cluster_stats(result, feature_cols, labels)
            if fig_stats:
                st.pyplot(fig_stats)

    elif algo == "Naive Bayes":
        st.subheader("Tham số cho thuật toán Naive Bayes")
        target = st.selectbox("Cột mục tiêu", [c for c in data_df.columns if c not in ("customer_id",)])
        numeric_cols = [c for c in data_df.select_dtypes(include=["number"]).columns if c != target]
        feature_cols = st.multiselect("Chọn đặc trưng (mặc định: cột số)", numeric_cols, default=numeric_cols)
        test_size = st.slider("Tỉ lệ test", 0.1, 0.5, 0.2)
        if st.button("Chạy Naive Bayes"):
            metrics, y_pred = run_nb(data_df, target=target, feature_columns=feature_cols, test_size=test_size)
            st.write("Kết quả:", metrics)

    elif algo == "Decision-Tree-CART":
        st.subheader("Tham số cho thuật toán Decision Tree (CART)")
        target = st.selectbox("Cột mục tiêu", [c for c in data_df.columns if c not in ("customer_id",)])
        numeric_cols = [c for c in data_df.select_dtypes(include=["number"]).columns if c != target]
        feature_cols = st.multiselect("Đặc trưng", numeric_cols, default=numeric_cols)
        max_depth = st.slider("Max depth", 1, 10, 5)
        min_split = st.slider("Min samples split", 2, 10, 2)
        if st.button("Chạy CART"):
            metrics = run_dt_cart(data_df, target=target, feature_columns=feature_cols, max_depth=max_depth, min_samples_split=min_split)
            st.write("Kết quả:", metrics)

    elif algo == "Decision-Tree-ID3":
        st.subheader("Tham số cho thuật toán Decision Tree (ID3 ~ entropy)")
        target = st.selectbox("Cột mục tiêu", [c for c in data_df.columns if c not in ("customer_id",)])
        numeric_cols = [c for c in data_df.select_dtypes(include=["number"]).columns if c != target]
        feature_cols = st.multiselect("Đặc trưng", numeric_cols, default=numeric_cols)
        max_depth = st.slider("Max depth", 1, 10, 5)
        min_split = st.slider("Min samples split", 2, 10, 2)
        if st.button("Chạy ID3"):
            metrics = run_dt_id3(data_df, target=target, feature_columns=feature_cols, max_depth=max_depth, min_samples_split=min_split)
            st.write("Kết quả:", metrics)

    elif algo == "Decision-Tree-C4.5":
        st.subheader("Tham số cho thuật toán Decision Tree (C4.5 mô phỏng)")
        target = st.selectbox("Cột mục tiêu", [c for c in data_df.columns if c not in ("customer_id",)])
        numeric_cols = [c for c in data_df.select_dtypes(include=["number"]).columns if c != target]
        feature_cols = st.multiselect("Đặc trưng", numeric_cols, default=numeric_cols)
        max_depth = st.slider("Max depth", 1, 10, 5)
        min_split = st.slider("Min samples split", 2, 10, 2)
        if st.button("Chạy C4.5"):
            metrics = run_dt_c45(data_df, target=target, feature_columns=feature_cols, max_depth=max_depth, min_samples_split=min_split)
            st.write("Kết quả:", metrics)

    elif algo == "Apriori":
        st.subheader("Tham số cho thuật toán Apriori")
        min_sup = st.slider("Min support", 0.01, 0.5, 0.02)
        min_conf = st.slider("Min confidence", 0.1, 1.0, 0.3)
        top_k = st.slider("Top-k luật", 5, 50, 10)

        if st.button("Chạy Apriori"):
            frequent, rules = run_apriori(data_df, min_support=min_sup, min_confidence=min_conf, top_k=top_k)
            st.write("Tập phổ biến thỏa min-support:", min_sup)
            st.dataframe(frequent.head(20))
            st.write("Luật liên kết (top-k):")
            st.dataframe(rules)

    elif algo == "Rough Set":
        st.subheader("Rút gọn thuộc tính (đơn giản)")
        target = st.selectbox("Cột mục tiêu", [c for c in data_df.columns if c not in ("customer_id",)])
        max_features = st.slider("Số thuộc tính tối đa", 1, 10, 5)
        if st.button("Chạy rút gọn"):
            kept = reduce_attributes(data_df.dropna(), target=target, max_features=max_features)
            st.success(f"Tập rút gọn là: {kept}")

    else:
        st.info("Chọn thuật toán ở sidebar để chạy.")


if __name__ == "__main__":
    main()


