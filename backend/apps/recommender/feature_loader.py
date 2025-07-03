# backend/recommender/features_loader.py

import pandas as pd

# Load on import— we can swap this for a cached DB or Redis lookup later

FEATURES_DF = pd.read_csv("data/processed/oulad_user_features.csv", index_col="id_student")

def load_features_for_student(student_id: int) -> dict | None:
    """
    Return the feature dict for one student, or None if not found.
    """
    try:
        row = FEATURES_DF.loc[student_id]
    except KeyError:
        return None
    # .to_dict() turns the Series into { column_name: value, … }
    return row.to_dict()
