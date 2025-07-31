import pickle
import os
import pandas as pd
from features.feature_extraction import extract_features


# Dynamically resolve the model path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "xgb_best_media_type_model.pkl")

# Load the model once at import time
with open(MODEL_PATH, "rb") as f:
    xgb_model = pickle.load(f)

def predict_best_modality(query, learner_level="intermediate"):
    """
    Predicts the best content type (video, book, course) using a trained XGBoost classifier.

    Args:
        query (str): User's search or topic query
        learner_level (str): 'beginner', 'intermediate', 'advanced'

    Returns:
        str: One of ['video', 'book', 'course']
    """
    features = extract_features(query, learner_level)
    df_feat = pd.DataFrame([features])
    # Map integer class → string label
    label_map = {0: "video", 1: "book", 2: "course"}
    pred = xgb_model.predict(df_feat)[0]
    return label_map[int(pred)]

    #pred = xgb_model.predict(df_feat)[0]
    #return pred
