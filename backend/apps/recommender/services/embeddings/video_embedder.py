import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

# ---------------------------
# Config
# ---------------------------
def load_config():
    load_dotenv()

    project_root = os.getenv("PROJECT_ROOT", os.getcwd())

    # INPUT (produced by video_metadata_fetcher.py)
    in_csv = os.getenv(
        "VIDEO_META_CSV",
        os.path.join(project_root, "backend", "data", "interim_large", "videos_metadata.csv"),
    )

    # OUTPUTS
    out_dir = os.getenv(
        "VIDEO_EMB_OUT",
        os.path.join(project_root, "backend", "data", "processed"),
    )
    os.makedirs(out_dir, exist_ok=True)
    out_meta = os.getenv(
        "VIDEO_EMBED_META_CSV",
        os.path.join(out_dir, "videos_metadata_clean.csv"),
    )
    out_npy = os.getenv(
        "VIDEO_EMBED_NPY",
        os.path.join(out_dir, "videos_embeddings.npy"),
    )

    # Model + batching
    model_name = os.getenv("VIDEO_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = int(os.getenv("VIDEO_EMBED_BATCH", "128"))

    return {
        "PROJECT_ROOT": project_root,
        "IN_CSV": in_csv,
        "OUT_DIR": out_dir,
        "OUT_META": out_meta,
        "OUT_NPY": out_npy,
        "MODEL_NAME": model_name,
        "BATCH_SIZE": batch_size,
    }

# ---------------------------
# Text utilities
# ---------------------------
WS_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = WS_RE.sub(" ", s).strip().lower()
    return s

def build_text_for_embedding(row: pd.Series) -> str:
    # Prefer richer text (title + description + tags + channel)
    parts = [
        clean_text(row.get("title", "")),
        clean_text(row.get("description", "")),
        clean_text(str(row.get("tags", "")).replace("|", " ")),
        clean_text(row.get("channel_title", "")),
    ]
    return " ".join([p for p in parts if p])

# ---------------------------
# Embedding
# ---------------------------
def load_model(name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Model] Loading '{name}' on device: {device}")
    model = SentenceTransformer(name, device=device)
    return model

def embed_corpus(model: SentenceTransformer, texts, batch_size: int = 128) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    # Save as float32 to reduce size (Sentence-Transformers already returns float32)
    return emb.astype(np.float32, copy=False)

# ---------------------------
# IO helpers
# ---------------------------
def read_input_csv(path: str) -> pd.DataFrame:
    print(f"[IO] Reading input: {path}")
    df = pd.read_csv(path)
    # Basic sanity: keep rows with video_id and title or description
    before = len(df)
    df = df.dropna(subset=["video_id"])
    # Deduplicate by video_id (keep the first occurrence)
    df = df.drop_duplicates(subset=["video_id"], keep="first")
    after = len(df)
    if after < before:
        print(f"[Clean] Dropped {before - after} rows due to missing/duplicate video_id.")
    return df

def write_outputs(df_meta: pd.DataFrame, embeddings: np.ndarray, meta_out: str, npy_out: str):
    print(f"[IO] Writing metadata CSV: {meta_out}")
    Path(os.path.dirname(meta_out)).mkdir(parents=True, exist_ok=True)
    df_meta.to_csv(meta_out, index=False)

    print(f"[IO] Writing embeddings NPY: {npy_out}  (shape={embeddings.shape})")
    Path(os.path.dirname(npy_out)).mkdir(parents=True, exist_ok=True)
    np.save(npy_out, embeddings)
    print("[Done]")

# ---------------------------
# Main
# ---------------------------
def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Embed YouTube video metadata into vectors.")
    parser.add_argument("--in_csv", default=cfg["IN_CSV"], help="Input metadata CSV (from fetcher).")
    parser.add_argument("--out_meta", default=cfg["OUT_META"], help="Output cleaned metadata CSV.")
    parser.add_argument("--out_npy", default=cfg["OUT_NPY"], help="Output embeddings .npy.")
    parser.add_argument("--model", default=cfg["MODEL_NAME"], help="Sentence-Transformers model name.")
    parser.add_argument("--batch_size", type=int, default=cfg["BATCH_SIZE"], help="Batch size for encoding.")
    args = parser.parse_args()

    df = read_input_csv(args.in_csv)

    # Build deterministic ordering by video_id to keep alignment stable across runs
    if "video_id" in df.columns:
        df = df.sort_values("video_id").reset_index(drop=True)

    # Construct text_for_embedding
    print("[Prep] Building text_for_embedding...")
    df["text_for_embedding"] = df.apply(build_text_for_embedding, axis=1)

    # Keep a lean metadata view (add/adjust as you need)
    keep_cols = [
        "video_id", "title", "description", "channel_id", "channel_title",
        "published_at", "view_count", "like_count", "comment_count",
        "tags", "topic_categories", "duration_seconds", "definition",
        "default_language", "default_audio_language", "source_query",
        "text_for_embedding",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_meta = df[keep_cols].copy()

    # Embed
    model = load_model(args.model)
    texts = df_meta["text_for_embedding"].fillna("").tolist()
    print(f"[Embed] Encoding {len(texts)} rows...")
    embeddings = embed_corpus(model, texts, batch_size=args.batch_size)

    # Persist
    write_outputs(df_meta, embeddings, args.out_meta, args.out_npy)


    # ---- write an id→row map like the books pipeline ----
    map_out = os.getenv("VIDEO_EMBED_MAP",
        os.path.join(cfg["OUT_DIR"], "video_embeddings_map.parquet")
    )

    map_df = pd.DataFrame({
        "video_id": df_meta["video_id"].values,
        "row_index": np.arange(len(df_meta), dtype=np.int32),
        "embedding_dim": embeddings.shape[1],
    })
    try:
        map_df.to_parquet(map_out, index=False)  # requires pyarrow or fastparquet
        print(f"[IO] Wrote map parquet: {map_out} (rows={len(map_df)})")
    except Exception as e:
        # Fallback to CSV if parquet writer isn't available
        fallback = os.path.splitext(map_out)[0] + ".csv"
        map_df.to_csv(fallback, index=False)
        print(f"[IO] Parquet failed ({e}); wrote CSV map instead: {fallback}")

if __name__ == "__main__":
    main()

