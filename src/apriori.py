from typing import Tuple
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(
    transactions_df: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.4,
    top_k: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tạo ma trận one-hot từ giao dịch, sinh tập phổ biến và luật liên kết.
    
    Args:
        transactions_df: DataFrame chứa 'Customer Reference ID' và 'Item Purchased'
        min_support: Ngưỡng hỗ trợ tối thiểu
        min_confidence: Ngưỡng tin cậy tối thiểu
        top_k: Số luật tốt nhất trả về
        
    Returns:
        Tuple[frequent_itemsets, rules]
    """
    # Xác định tên cột
    if 'Customer Reference ID' in transactions_df.columns and 'Item Purchased' in transactions_df.columns:
        customer_col, item_col = 'Customer Reference ID', 'Item Purchased'
    elif 'transaction_id' in transactions_df.columns and 'item' in transactions_df.columns:
        customer_col, item_col = 'transaction_id', 'item'
    else:
        raise ValueError("Không tìm thấy cột cần thiết")

    # Tạo danh sách giao dịch
    grouped = transactions_df.groupby(customer_col)[item_col].apply(list).tolist()
    grouped = [items for items in grouped if items]  # Loại bỏ empty
    
    if not grouped:
        return _empty_results()
    
    # Điều chỉnh tham số cho dataset nhỏ
    min_support = _adjust_min_support(min_support, len(grouped))
    
    # Tạo ma trận One-Hot và chạy Apriori
    te = TransactionEncoder()
    ohe = pd.DataFrame(te.fit_transform(grouped), columns=te.columns_)
    
    frequent = apriori(ohe, min_support=min_support, use_colnames=True)
    
    if frequent.empty:
        return frequent, _empty_rules()
    
    # Tạo luật liên kết
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by=["lift", "confidence"], ascending=False).head(top_k)
    
    # Format kết quả
    formatted_rules = _format_rules(rules)
    
    return frequent.sort_values("support", ascending=False), formatted_rules


def _empty_results() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về kết quả rỗng."""
    empty_frequent = pd.DataFrame(columns=['itemsets', 'support'])
    empty_rules = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
    return empty_frequent, empty_rules


def _empty_rules() -> pd.DataFrame:
    """Trả về DataFrame luật rỗng."""
    return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])


def _adjust_min_support(min_support: float, num_transactions: int) -> float:
    """Điều chỉnh min_support cho dataset nhỏ."""
    if num_transactions < 50:
        min_support = max(0.02, min_support * 0.3)
    
    # Kiểm tra min_support
    min_transactions = max(2, int(num_transactions * min_support))
    if min_transactions > num_transactions * 0.8:
        min_support = 0.02
    
    return min_support


def _format_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Format luật liên kết thành dạng dễ đọc."""
    def _fmt(itemset):
        return ", ".join(sorted(list(itemset)))
    
    result = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    result["antecedents"] = result["antecedents"].apply(_fmt)
    result["consequents"] = result["consequents"].apply(_fmt)
    
    return result
