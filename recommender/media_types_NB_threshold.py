import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

# Load & prepare data
df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')

count_cols = ['course_count', 'reading_count', 'video_count']
df['total_count'] = df[count_cols].sum(axis=1)
for c in count_cols:
    df[c.replace('_count', '_prop')] = df[c] / df['total_count']
df['top_bucket'] = (
    df[['course_prop','reading_prop','video_prop']]
      .idxmax(axis=1)
      .str.replace('_prop','')
)

X = df[count_cols]
y = df['top_bucket']

#  Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#  Fit NB with ×10 boosted reading prior
#  (recompute priors on the training split)
base_nb = MultinomialNB(alpha=1.0)
base_nb.fit(X_train, y_train)
classes = base_nb.classes_
emp = y_train.value_counts(normalize=True).reindex(classes).fillna(0).values
rid = list(classes).index('reading')

priors = emp.copy()
priors[rid] *= 10
priors /= priors.sum()

clf = MultinomialNB(alpha=1.0, class_prior=priors)
clf.fit(X_train, y_train)

# Get posterior probabilities on the test set
probs = clf.predict_proba(X_test)
# find the index of reading in classes_
idx_r = rid

# Sweep thresholds and compute F1 on the reading class
thresholds = np.linspace(0.001, 0.1, 20)  # from 0.1% to 10%
results = []
for T in thresholds:
    # predict reading if P(reading) > T, else fall back to argmax of the other two
    other_idxs = [i for i in range(len(classes)) if i != idx_r]
    # indices of the remaining classes
    def predict_with_thresh(p_row):
        if p_row[idx_r] > T:
            return 'reading'
        # otherwise pick whichever of the others has higher posterior
        return classes[other_idxs[np.argmax(p_row[other_idxs])]]
    
    y_pred = [predict_with_thresh(p) for p in probs]
    f1 = f1_score(y_test, y_pred, labels=['reading'], average='macro', zero_division=0)
    results.append((T, f1))

# Display the results
print("Threshold  → Reading F1")
print("------------------------")
for T, f1 in results:
    print(f"{T:>8.4f}  → {f1:.3f}")

# 7. Identify best T
best_T, best_f1 = max(results, key=lambda x: x[1])
print(f"\nBest threshold: {best_T:.4f}  (Reading F1 = {best_f1:.3f})")
