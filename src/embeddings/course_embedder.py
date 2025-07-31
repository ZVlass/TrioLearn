import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

#  Load the cleaned combined catalog

catalog_path = "./data/interim/courses_metadata.csv"
df = pd.read_csv(catalog_path)

# Check that text_for_embedding exists
if 'text_for_embedding' not in df.columns:
    raise KeyError("Column 'text_for_embedding' not found in DataFrame.")

print("Loaded combined courses catalog with shape:", df.shape)

# Choose a Sentence-BERT model
model_name = 'all-MiniLM-L6-v2'  
model = SentenceTransformer(model_name)

# Encode in batches
texts = df['text_for_embedding'].fillna("").astype(str).tolist()
print("Encoding", len(texts), "courses into embeddings...")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)  # returns a numpy array

# Save embeddings to disk
enbeddings_dir = "./data/embeddings"
emb_path = os.path.join(enbeddings_dir, "course_embeddings.npy")
np.save(emb_path, embeddings)
print("Saved embeddings to:", emb_path)




