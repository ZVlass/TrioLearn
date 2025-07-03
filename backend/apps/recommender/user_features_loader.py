

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

# build the path to data/processed/oulad_user_features.csv
FEATURES_PATH = BASE_DIR / "data" / "processed" / "oulad_user_features.csv"

FEATURES_DF = pd.read_csv(FEATURES_PATH, index_col="id_student")

def load_features_for_student(student_id: int) -> dict | None:
    try:
        row = FEATURES_DF.loc[student_id]
    except KeyError:
        return None
    return row.to_dict()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python user_features_loader.py <student_id>")
        sys.exit(1)
    sid = int(sys.argv[1])
    feats = load_features_for_student(sid)
    if feats is None:
        print(f"Student {sid} not found")
        sys.exit(2)
    # pretty‐print the dict
    import json
    print(json.dumps(feats, indent=2))

