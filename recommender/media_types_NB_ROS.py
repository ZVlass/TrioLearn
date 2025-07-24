import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

# 1. Load data
df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')

# 2. Compute proportions & derive top_bucket
count_cols = ['course_count', 'reading_count', 'video_count']
df['total_count'] = df[count_cols].sum(axis=1)
for c in count_cols:
    df[c.replace('_count', '_prop')] = df[c] / df['total_count']
df['top_bucket'] = df[['course_prop','reading_prop','video_prop']].idxmax(axis=1).str.replace('_prop','')

# 3. Prepare X & y
X = df[count_cols]
y = df['top_bucket']

# 4. Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== Approach A: Oversample the minority class =====
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

mnb_os = MultinomialNB(alpha=1.0)
mnb_os.fit(X_res, y_res)
y_pred_os = mnb_os.predict(X_test)

print("=== Oversampling Minority Class ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_os):.3f}")
print(classification_report(y_test, y_pred_os))


# ===== Approach B: Manually set priors =====
# Compute empirical priors then boost the reading prior
empirical_priors = y_train.value_counts(normalize=True).reindex(mnb_os.classes_).fillna(0).values
# e.g. multiply reading prior by 10, then renormalize
priors = empirical_priors.copy()
idx_reading = list(mnb_os.classes_).index('reading')
priors[idx_reading] *= 10
priors /= priors.sum()

mnb_prior = MultinomialNB(alpha=1.0, class_prior=priors)
mnb_prior.fit(X_train, y_train)
y_pred_pr = mnb_prior.predict(X_test)

print("=== Boosted Reading Prior ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_pr):.3f}")
print(classification_report(y_test, y_pred_pr))


# ===== Example Prediction =====
new_user = pd.DataFrame([{'course_count':120, 'reading_count':30, 'video_count':10}])
print("--- New User Recommendations ---")
print("Oversampled NB →", mnb_os.predict(new_user)[0],
      mnb_os.predict_proba(new_user).round(4))
print("Prior-boosted NB →", mnb_prior.predict(new_user)[0],
      mnb_prior.predict_proba(new_user).round(4))
