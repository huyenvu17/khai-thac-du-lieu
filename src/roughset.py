from typing import List
import pandas as pd


def reduce_attributes(
    df: pd.DataFrame,
    target: str,
    max_features: int = 5,
) -> List[str]:
    """Ý nghĩa: chọn các cột có tương quan tuyệt đối cao nhất với target (nếu số),
    hoặc chọn theo mutual_info cho mục tiêu phân loại.
    """
    y = df[target]
    X = df.drop(columns=[target])

    # Ưu tiên đặc trưng số đơn giản.
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) > 0 and pd.api.types.is_numeric_dtype(y):
        corrs = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
        return corrs.head(max_features).index.tolist()

    # Trường hợp phân loại: dùng mutual information
    try:
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import OrdinalEncoder
        X_enc = X.copy()
        # Mã hóa danh mục đơn giản nếu có
        cat_cols = X_enc.select_dtypes(exclude=["number"]).columns.tolist()
        if cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_enc[cat_cols] = enc.fit_transform(X_enc[cat_cols].astype(str))
        mi = mutual_info_classif(X_enc, y, discrete_features="auto")
        scores = pd.Series(mi, index=X_enc.columns).sort_values(ascending=False)
        return scores.head(max_features).index.tolist()
    except Exception:
        # Fallback: trả về cột số đầu tiên
        return numeric_cols[:max_features]


