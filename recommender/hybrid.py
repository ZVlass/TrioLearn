"""
Hybrid recommendation models: weighted-sum, stacking, LightFM.
"""
import numpy as np


def weighted_score(user_emb: np.ndarray, item_embs: np.ndarray, cf_scores: np.ndarray, topic_scores: np.ndarray,
                   alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2) -> np.ndarray:
    
    """Compute weighted-sum hybrid score matrix."""
    
    # user_emb: (U, D), item_embs: (I, D)
    cos_scores = item_embs.dot(user_emb.T)  # shape (I, U)
    return alpha * cos_scores + beta * cf_scores + gamma * topic_scores