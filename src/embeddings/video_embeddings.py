from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os

input_path = "./data/interim/ml_videos.csv"
print("Absolute path:", os.path.abspath(input_path))
print("File exists?", os.path.exists(input_path))

def clean_text(text):
    return text.replace('\n', ' ').replace('\r', '').strip().lower() if isinstance(text, str) else ''

def embed_youtube_df(df, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    df['text'] = (df['title'].fillna('') + " " + df['description'].fillna('')).apply(clean_text)
    embeddings = model.encode(df['text'].tolist(), normalize_embeddings=True)
    return df.drop(columns=['text']), embeddings  # return cleaned df and raw embedding matrix


input_path = "./data/interim/ml_videos.csv"
out_metadata = "./data/processed/ml_videos_metadata.csv"
out_embeddings = "./data/processed/ml_videos_embeddings.npy"


print(f"Loading: {input_path}")
df = pd.read_csv(input_path)

print("Generating embeddings...")
df_clean, embeddings = embed_youtube_df(df)

print(f"Saving metadata to: {out_metadata}")
df_clean.to_csv(out_metadata, index=False)

print(f"Saving embeddings to: {out_embeddings}")
np.save(out_embeddings, embeddings)

print("Done.")
