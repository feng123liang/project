import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_anomaly_model(y_true, anomaly_scores, K=50):
    """
    Evaluate anomaly detection model with ROC-AUC, PR-AUC, and Precision@K.
    
    y_true: array-like, binary labels (0 = normal, 1 = anomaly)
    anomaly_scores: array-like, higher values = more anomalous
    K: int, top-K threshold for Precision@K
    """
    roc_auc = roc_auc_score(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)
    
    idx_topK = np.argsort(-anomaly_scores)[:K]
    precision_at_k = np.mean(y_true[idx_topK])
    
    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        f"Precision@{K}": precision_at_k
    }

def precision_at_k(y_true, scores, k=10):
    """计算Precision@K"""
    if len(scores) < k:
        k = len(scores)
    top_k_indices = np.argsort(scores)[-k:]  # 分数越高越异常
    return y_true.iloc[top_k_indices].mean()