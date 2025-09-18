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
    """Naive Bayes đơn giản cho người mới: dùng GaussianNB trên cột số.
    Nếu truyền feature_columns thì dùng các cột đó; nếu không sẽ lấy cột số trừ target.
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix
    from .utils import train_test_split_df
    from .utils import select_columns
    from .utils import compute_classification_metrics

    if feature_columns:
        used_df = df[feature_columns + [target]].dropna()
    else:
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        numeric_cols = [c for c in numeric_cols if c != target]
        used_df = df[numeric_cols + [target]].dropna()
        feature_columns = numeric_cols

    X_train, X_test, y_train, y_test = train_test_split_df(
        used_df, target=target, test_size=test_size, random_state=random_state
    )

    model = GaussianNB(var_smoothing=1e-9)
    # Laplace smoothing tương đương alpha áp dụng tự nhiên hơn trong Multinomial/Bernoulli.
    # Với GaussianNB, ta vẫn giữ tham số mặc định cho đơn giản.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    metrics = compute_classification_metrics(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["features"] = feature_columns
    return metrics, pd.Series(y_pred, index=y_test.index, name="y_pred")


