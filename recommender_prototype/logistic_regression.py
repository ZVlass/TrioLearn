import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the OULAD media profiles dataset
df = pd.read_csv('./data/processed/oulad_media_profiles.csv')

# Create target variable: predominant media profile
df['profile'] = df[['course_prop', 'other_prop', 'reading_prop', 'video_prop']].idxmax(axis=1).str.replace('_prop', '')

# Features and target
df_features = df[['course_count', 'other_count', 'reading_count', 'video_count']]
X = df_features
y = df['profile']

# Encode target labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Evaluate performance
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f}")

# Classification report
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("Classification Report:\n", report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Cross-validation
cv_scores = cross_val_score(model, scaler.fit_transform(X), y_enc, cv=5, scoring='accuracy')
print("5-fold CV accuracy scores:", cv_scores)
print(f"Mean CV accuracy: {cv_scores.mean():.3f}")

# Plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm)
ax.set_xticks(range(len(le.classes_)))
ax.set_yticks(range(len(le.classes_)))
ax.set_xticklabels(le.classes_)
ax.set_yticklabels(le.classes_)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
