"""
Evaluation metrics: Precision@K, Recall@K, AUC.
"""


import numpy as np
from sklearn.metrics import roc_auc_score


def precision_at_k(recs: list, true: int, k: int) -> float:
    return float(true in recs[:k]) / k


def recall_at_k(recs: list, true: int, k: int) -> float:
    return float(true in recs[:k])


def auc_score(user_profile: np.ndarray, item_embs: np.ndarray, true_idx: int, negative_idxs: np.ndarray) -> float:
    y_true = [1] + [0] * len(negative_idxs)
    # scores: first true, then negatives
    true_score = item_embs[true_idx].dot(user_profile)
    neg_scores = item_embs[negative_idxs].dot(user_profile)
    y_scores = [true_score] + neg_scores.tolist()
    return roc_auc_score(y_true, y_scores)


def evaluate_all(user_profiles: np.ndarray, item_embs: np.ndarray, test_pairs: list, K: int = 10) -> dict:
    """Compute average Precision@K, Recall@K, AUC over all test users."""
    precs, recs, aucs = [], [], []
    for u_idx, true_item in test_pairs:
        # placeholder recommend function
        recs_list = []  # TODO: generate rec list of indices
        negs = np.random.choice([i for i in range(item_embs.shape[0]) if i != true_item], size=100, replace=False)
        precs.append(precision_at_k(recs_list, true_item, K))
        recs.append(recall_at_k(recs_list, true_item, K))
        aucs.append(auc_score(user_profiles[u_idx], item_embs, true_item, negs))
    return {'Precision@K': np.mean(precs), 'Recall@K': np.mean(recs), 'AUC': np.mean(aucs)}