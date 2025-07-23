import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load media profile data
df = pd.read_csv('./data/processed/oulad_media_profiles_full.csv')


# Create the target label: user's top preferred media type
count_cols = ['course_count', 'reading_count', 'video_count', 'other_count']
df['top_media'] = df[count_cols].idxmax(axis=1).str.replace('_count', '')

# Select features (numeric counts and proportions)
feature_cols = count_cols + ['course_prop', 'reading_prop', 'video_prop', 'other_prop']
X = df[feature_cols]
y = df['top_media']

# Encode target labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42
)

# Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

#  Predict and evaluate
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Preview a sample of features and labels
print("\nSample of feature values and corresponding top media labels:")

print(df[feature_cols + ['top_media']].head())

print("\nLabel encoding mapping:")
for cls, code in zip(le.classes_, range(len(le.classes_))):
    print(f"  {cls}: {code}")
