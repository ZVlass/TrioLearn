import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
df = pd.read_csv('./data/processed/oulad_media_profiles_refined.csv')  # adjust path if needed

# 2. Derive the target: which proportion (course/other/reading/video) is largest?
prop_cols = ['course_prop', 'reading_prop', 'video_prop']
# idxmax gives the column name; strip off '_prop' to get the bucket label
df['top_bucket'] = (
    df[prop_cols]
      .idxmax(axis=1)
      .str.replace('_prop', '', regex=False)
)

# 3. Prepare features (X) and target (y)
feature_cols = ['course_count', 'reading_count', 'video_count']
X = df[feature_cols]
y = df['top_bucket']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # preserves bucket distribution in train/test
)

# 5. Train a Multinomial Naive Bayes (counts → discrete NB)
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Evaluate on the test set
y_pred = model.predict(X_test)
print(f'Overall accuracy: {accuracy_score(y_test, y_pred):.3f}\n')
print('Detailed classification report:')
print(classification_report(y_test, y_pred))

# 7. Example: predict recommendation for a new user profile
new_user = pd.DataFrame([{
    'course_count': 120,
    'reading_count': 30,
    'video_count': 10
}])
pred_bucket = model.predict(new_user)[0]
probs = model.predict_proba(new_user)[0]
class_probs = dict(zip(model.classes_, probs))

print(f'\nFor new user counts {new_user.iloc[0].to_dict()}:')
print(f'→ Recommended bucket: {pred_bucket}')
print(f'→ Probability per bucket: {class_probs}')
