import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score


# Load data
df = pd.read_csv('./data/processed/oulad_media_profiles_refined_balanced.csv')

# Define count columns and compute total
count_cols = ['course_count', 'reading_count', 'video_count']
df['total_count'] = df[count_cols].sum(axis=1)

# Compute proportions
for c in count_cols:
    df[c.replace('_count', '_prop')] = df[c] / df['total_count']

# Derive the target bucket
prop_cols = ['course_prop', 'reading_prop', 'video_prop']
df['top_bucket'] = df[prop_cols].idxmax(axis=1).str.replace('_prop', '')

# Features and target
X = df[count_cols]
y = df['top_bucket']

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Compute empirical priors for NB classes
mnb = MultinomialNB(alpha=1.0)  # dummy to get classes_
mnb.fit(X_train, y_train)
classes = mnb.classes_
emp_priors = y_train.value_counts(normalize=True).reindex(classes).fillna(0).values
reading_idx = list(classes).index('reading')

# 4. Grid-search over reading-prior multipliers
factors = [5, 10, 20, 50, 100, 200, 500]
results = []

for factor in factors:
    # Boost the reading prior
    priors = emp_priors.copy()
    priors[reading_idx] *= factor
    priors /= priors.sum()
    
    # Train with custom priors
    clf = MultinomialNB(alpha=1.0, class_prior=priors)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate F1 on the reading class
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred,
                  labels=['reading'],
                  average='macro',
                  zero_division=0)
    
    results.append((factor, f1))

# 5. Sort and display the results
results.sort(key=lambda x: x[1], reverse=True)

print("Reading-Prior Multiplier → Reading F1")
print("--------------------------------------")
for factor, f1 in results:
    print(f"{factor:>7} → {f1:.3f}")

# Optionally: print the best factor
best_factor, best_f1 = results[0]
print(f"\nBest multiplier: ×{best_factor}  (Reading F1 = {best_f1:.3f})")