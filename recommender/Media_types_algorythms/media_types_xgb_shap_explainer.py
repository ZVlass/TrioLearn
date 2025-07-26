import shap
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# ------------------ Load Model and Encoder ------------------ #
model_path = r"C:\Users\jvlas\source\repos\TrioLearn\outputs\models\xgb_smote_model.pkl"
encoder_path = r"C:\Users\jvlas\source\repos\TrioLearn\outputs\models\label_encoder.pkl"

pipeline_model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# ------------------ Load Processed Data ------------------ #
df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')
count_cols = ['course_count', 'reading_count', 'video_count']
X = df[count_cols].copy()

# ------------------ Extract XGBClassifier ------------------ #
if hasattr(pipeline_model, "named_steps") and "xgb" in pipeline_model.named_steps:
    xgb_model = pipeline_model.named_steps["xgb"]
else:
    raise ValueError("XGBoost model not found in pipeline.")

# ------------------ SHAP Explanations ------------------ #
explainer = shap.Explainer(xgb_model, X)
shap_values = explainer(X)

# Global Summary Plot
shap.summary_plot(shap_values, X)

# Optional: Individual Force Plot
sample_idx = 10  # change to any row index
shap.initjs()
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx].values,
    features=X.iloc[sample_idx],
    feature_names=X.columns
)
