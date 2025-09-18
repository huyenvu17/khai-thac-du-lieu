import pandas as pd
from typing import List, Tuple


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if not columns:
        return df
    return df[columns]


def train_test_split_df(
    df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target])
    y = df[target]
    
    # Kiểm tra xem có thể dùng stratified split không
    # Cần ít nhất 2 mẫu cho mỗi class để stratified split hoạt động
    can_stratify = True
    if y.nunique() > 1:
        min_class_count = y.value_counts().min()
        if min_class_count < 2:
            can_stratify = False
    
    stratify_param = y if can_stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_param)


def scale_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if not columns:
        return df
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled


def compute_classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1)}

