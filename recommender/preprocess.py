
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_user_features(path: str) -> pd.DataFrame:
    """Load and preprocess OULAD user features CSV."""
    df = pd.read_csv(path)
    # TODO: handle missing/infinite, scale features
    return df


def load_course_metadata(path: str) -> pd.DataFrame:
    """Load course metadata (titles, descriptions, etc.)."""
    df = pd.read_csv(path)
    # TODO: text cleaning, embedding inputs
    return df

def preprocess_user_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_cols = [c for c in df.columns if c != 'id_student']
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
    df[feature_cols] = StandardScaler().fit_transform(df[feature_cols])
    return df