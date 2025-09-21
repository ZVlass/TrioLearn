import shap
import joblib
import pandas as pd
import os


model_path = r"C:\Users\jvlas\source\repos\TrioLearn\outputs\models\xgb_smote_model.pkl"
encoder_path = r"C:\Users\jvlas\source\repos\TrioLearn\outputs\models\label_encoder.pkl"
data_path = r"C:\Users\jvlas\source\repos\TrioLearn\data\processed\oulad_media_profiles_refined_balanced.csv"
output_html = r"C:\Users\jvlas\source\repos\TrioLearn\outputs\models\shap_force_learner10.html"
learner_index = 10 


pipeline_model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)
df = pd.read_csv(data_path)
X = df[['course_count', 'reading_count', 'video_count']]

xgb_model = pipeline_model.named_steps['xgb']

explainer = shap.Explainer(xgb_model, X)
shap_values = explainer(X)

# Select one learner
shap_value = shap_values[learner_index]
features = X.iloc[learner_index]

# Save force plot to HTML
shap.initjs()
html = shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=shap_value.values,
    features=features,
    feature_names=X.columns,
    matplotlib=False,
    show=False
)

# Save HTML file
with open(output_html, "w") as f:
    f.write(html.html())

print(f"\n SHAP force plot saved to:\n{output_html}")
