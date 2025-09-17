
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from django.conf import settings
from apps.recommender.feature_extraction import extract_features  # returns the schema we trained on

BASE_DIR = Path(settings.BASE_DIR)

MODEL_PATH = os.getenv(
    "XGB_MODEL_PATH",
    str(BASE_DIR / "outputs" / "models" / "xgb_best_model.pkl")
)
ENC_PATH = os.getenv(
    "XGB_LABEL_ENCODER_PATH",
    str(BASE_DIR / "outputs" / "models" / "label_encoder.pkl")
)

# Load model (can be bare XGBClassifier or an sklearn/imb-learn Pipeline)
with open(MODEL_PATH, "rb") as f:
    xgb_model = pickle.load(f)

# Optional: LabelEncoder (only used for mapping if model.classes_ are numeric)
label_encoder = None
if os.path.exists(ENC_PATH):
    try:
        label_encoder = joblib.load(ENC_PATH)
    except Exception:
        label_encoder = None

def _normalize_to_class_index(raw_pred):
    """
    Accepts many shapes:
      - scalar int/str             -> return int or str
      - 1D array of length 1      -> class id
      - 1D vector probs/one-hot   -> argmax
      - 2D (1, n_classes) probs   -> argmax
    """
    arr = np.asarray(raw_pred)
    # Model might return already-decoded string label
    if arr.ndim == 0:
        val = arr.item()
        try:
            return int(val)
        except Exception:
            return str(val)
    if arr.ndim == 1:
        if arr.size == 1:
            val = arr[0]
            try:
                return int(val)
            except Exception:
                return str(val)
        # vector -> argmax
        return int(np.argmax(arr))
    if arr.ndim == 2:
        # e.g., (1, n_classes)
        return int(np.argmax(arr[0]))
    # Fallback: return string representation
    return str(arr)

def _label_from_pred(pred_class_id: int) -> str:
    # Prefer model.classes_ if they’re strings
    classes = getattr(xgb_model, "classes_", None)
    if classes is not None and any(isinstance(c, str) for c in classes):
        return str(classes[int(pred_class_id)])
    # Else try LabelEncoder
    if label_encoder is not None and getattr(label_encoder, "classes_", None) is not None:
        le_classes = list(label_encoder.classes_)
        if any(isinstance(c, str) for c in le_classes):
            return str(le_classes[int(pred_class_id)])
    # Final fallback (adjust if your order differs)
    return {0: "video", 1: "book", 2: "course"}.get(int(pred_class_id), "course")

def predict_best_modality(query: str, learner_level: str = "intermediate") -> str:
    feats = extract_features(query, learner_level)  # {'course_count','reading_count','video_count'}
    X = pd.DataFrame([feats])
    raw_pred = xgb_model.predict(X)
    cls_or_label = _normalize_to_class_index(raw_pred)
    # If the model already returned a string label, pass it through
    if isinstance(cls_or_label, str) and not cls_or_label.isdigit():
        return cls_or_label
    # Otherwise map the integer class id to a string
    return _label_from_pred(int(cls_or_label))
