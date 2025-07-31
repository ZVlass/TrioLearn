import os
import pandas as pd
import numpy as np


df_books = pd.read_csv("C:/Users/jvlas/source/repos/TrioLearn/data/interim/books_metadata.csv")

# Fill missing fields (avoid errors during string joining)
df_books.fillna("", inplace=True)

# Combine text fields into one string for embedding
df_books["text_for_embedding"] = (
    df_books["title"] + " " +
    df_books["description"] + " " +
    df_books["categories"]
)

# Load the pre-trained Sentence-BERT model

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
print(" Model loaded")

#  Compute embeddings (batch process for performance)
embeddings = model.encode(df_books["text_for_embedding"].tolist(), show_progress_bar=True)


# Store as separate columns
embedding_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])

# Concatenate with original metadata
df_books_embedded = pd.concat([df_books, embedding_df], axis=1)

interim_dir = os.path.join("data", "interim")

save_path = os.path.join(interim_dir, "books_with_embeddings.csv")
#df_books_embedded.to_csv(save_path, index=False)

print(f"Saved {len(df_books_embedded)} books with embeddings to:", save_path)

np.save("./data/processed/book_embeddings.npy", embeddings)