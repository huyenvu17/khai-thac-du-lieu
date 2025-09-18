from typing import Tuple
import pandas as pd


def run_apriori(
    transactions_df: pd.DataFrame,
    min_support: float = 0.02,
    min_confidence: float = 0.3,
    top_k: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tạo ma trận one-hot từ giao dịch (transaction_id, item), sinh tập phổ biến và luật.
    Trả về (frequent_itemsets, rules)."""
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules

    # Chuyển sang list giao dịch: [[item1, item2], ...]
    grouped = transactions_df.groupby("transaction_id")["item"].apply(list).tolist()
    te = TransactionEncoder()
    arr = te.fit_transform(grouped)
    ohe = pd.DataFrame(arr, columns=te.columns_)

    frequent = apriori(ohe, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    # Sắp xếp và lấy top-k
    rules = rules.sort_values(by=["lift", "confidence"], ascending=False).head(top_k)

    # Chỉ giữ cột cần thiết và chuyển frozenset -> chuỗi dễ đọc
    def _fmt(itemset):
        return ", ".join(sorted(list(itemset)))

    apriori_result_col = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    apriori_result_col["antecedents"] = apriori_result_col["antecedents"].apply(_fmt)
    apriori_result_col["consequents"] = apriori_result_col["consequents"].apply(_fmt)

    return frequent.sort_values("support", ascending=False), apriori_result_col


