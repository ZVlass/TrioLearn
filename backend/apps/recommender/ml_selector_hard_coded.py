import pickle
import os
import pandas as pd
from apps.recommender.feature_extraction import extract_features


#  Get path to the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../../../outputs/models/xgb_best_media_type_model.pkl")


# Load the model once
with open(os.path.abspath(MODEL_PATH), "rb") as f:
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
