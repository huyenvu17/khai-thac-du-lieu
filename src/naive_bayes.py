from typing import Dict, List, Tuple
import pandas as pd


def run_nb(
    df: pd.DataFrame,
    target: str,
    feature_columns: List[str] | None = None,
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dict, pd.Series]:
    """Naive Bayes đơn giản cho người mới: xử lý cả categorical và numerical features.
    Nếu truyền feature_columns thì dùng các cột đó; nếu không sẽ lấy cột số trừ target.
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import confusion_matrix
    from .utils import train_test_split_df
    from .utils import compute_classification_metrics

    if feature_columns:
        used_df = df[feature_columns + [target]].dropna()
    else:
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        numeric_cols = [c for c in numeric_cols if c != target]
        used_df = df[numeric_cols + [target]].dropna()
        feature_columns = numeric_cols

    # Xử lý categorical features
    processed_df = used_df.copy()
    label_encoders = {}
    
    for col in feature_columns:
        if used_df[col].dtype == 'object':  # Categorical column
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(used_df[col].astype(str))
            label_encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split_df(
        processed_df, target=target, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GaussianNB(var_smoothing=1e-9)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    metrics = compute_classification_metrics(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["features"] = feature_columns
    return metrics, pd.Series(y_pred, index=y_test.index, name="y_pred")


