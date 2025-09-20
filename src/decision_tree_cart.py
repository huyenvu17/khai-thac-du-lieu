from typing import Dict, List, Tuple
import pandas as pd


def run_dt_cart(
    df: pd.DataFrame,
    target: str,
    feature_columns: List[str] | None = None,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix
    from .utils import train_test_split_df, compute_classification_metrics

    if feature_columns:
        used_df = df[feature_columns + [target]].dropna()
    else:
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        numeric_cols = [c for c in numeric_cols if c != target]
        used_df = df[numeric_cols + [target]].dropna()
        feature_columns = numeric_cols

    # Xử lý categorical features
    processed_df = used_df.copy()
    
    for col in feature_columns:
        if used_df[col].dtype == 'object':  # Categorical column
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(used_df[col].astype(str))

    X_train, X_test, y_train, y_test = train_test_split_df(
        processed_df, target=target, test_size=test_size, random_state=random_state
    )

    model = DecisionTreeClassifier(
        criterion="gini", max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    metrics = compute_classification_metrics(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["features"] = feature_columns
    return metrics


