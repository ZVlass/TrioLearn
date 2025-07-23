import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

# Load the OULAD media profiles dataset
df = pd.read_csv('./data/processed/oulad_media_profiles.csv')

# Derive multi-class 'profile' target based on usage proportions
prop_cols = ['course_prop', 'other_prop', 'reading_prop', 'video_prop']
df[prop_cols] = df[prop_cols].fillna(0)
# Remove any rows where all proportions are zero
df = df[df[prop_cols].sum(axis=1) > 0].copy()
# Assign profile by the highest proportion and remove '_prop' suffix
df['profile'] = df[prop_cols].idxmax(axis=1).str.replace('_prop', '', regex=False)

# Prepare features X and target y
y = df['profile']
# Drop non-feature columns (identifiers and target)
X = df.drop(columns=['id_student'] + prop_cols + ['profile'])

# Select numeric features only
numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Resample training set to balance classes
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

# Scale features
def scale_data(X_train, X_eval):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    return X_train_scaled, X_eval_scaled

X_res_scaled, X_test_scaled = scale_data(X_res, X_test)

# Initialize model

models = {
    'LogisticRegression': LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=500),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate each model
results = {}
for name, clf in models.items():
    clf.fit(X_res_scaled, y_res)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    cv_scores = cross_val_score(
        clf, np.vstack([X_res_scaled, X_test_scaled]),
        np.hstack([y_res, y_test]), cv=5)
    results[name] = {
        'accuracy': acc,
        'classification_report': report,
        'confusion_matrix': cm,
        'cv_scores': cv_scores
    }

# Print results
for name, res in results.items():
    print(f"Model: {name}")
    print(f"Test Accuracy: {res['accuracy']:.3f}")
    print("Classification Report:")
    print(res['classification_report'])
    print("Confusion Matrix:")
    print(res['confusion_matrix'])
    print(f"5-fold CV accuracy: {res['cv_scores']}")
    print(f"Mean CV accuracy: {res['cv_scores'].mean():.3f}\n")

# Plot confusion matrices
fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
for ax, (name, res) in zip(axes.flatten(), results.items()):
    cm = res['confusion_matrix']
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
