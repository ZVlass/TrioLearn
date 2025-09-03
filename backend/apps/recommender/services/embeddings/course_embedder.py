import os, re, argparse, hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

# ---------- tiny helpers ----------
WS_RE = re.compile(r"\s+")
def clean(s):
    if not isinstance(s, str): return ""
    return WS_RE.sub(" ", s.replace("\r"," ").replace("\n"," ")).strip().lower()

def build_text_for_embedding(row: pd.Series) -> str:
    parts = [row.get(k, "") for k in ["title","description","skills","tags","provider","platform"]]
    return " ".join([clean(p) for p in parts if p])

def deterministic_course_id(df: pd.DataFrame) -> pd.Series:
    def normcol(name):
        if name in df.columns:
            return df[name].astype(str).map(clean)
        # return an empty series of the right length if column missing
        return pd.Series([""] * len(df), index=df.index)
    key = (
        normcol("title")    + "|" +
        normcol("provider") + "|" +
        normcol("level")    + "|" +
        normcol("platform") + "|" +
        normcol("duration")
    )
    return key.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:16])

# ---------- IO ----------
def read_courses(path: str) -> (pd.DataFrame, str):
    print(f"[IO] Reading: {path}")
    df = pd.read_csv(path)

    # tidy stray index column
    if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])

    # canonize common fields from various sources
    if "title" not in df.columns and "Title" in df.columns: df["title"] = df["Title"]
    if "description" not in df.columns and "course_description" in df.columns:
        df["description"] = df["course_description"]
    if "provider" not in df.columns and "Organization" in df.columns:
        df["provider"] = df["Organization"]
    if "skills" not in df.columns and "Skills" in df.columns: df["skills"] = df["Skills"]
    if "level" not in df.columns and "Difficulty" in df.columns:
        df["level"] = df["Difficulty"]
    if "platform" not in df.columns and "Type" in df.columns:
        df["platform"] = df["Type"]
    if "duration" not in df.columns:
        if "Duration" in df.columns:
            df["duration"] = df["Duration"]

    # choose/generate an id
    id_col = next((c for c in ["course_id","global_id","id"] if c in df.columns), None)
    if not id_col:
        df["course_id"] = deterministic_course_id(df)
        id_col = "course_id"
        print(f"[ID] Generated '{id_col}' deterministically from key fields.")

    # hygiene + stable order
    before = len(df)
    df = df.dropna(subset=[id_col]).drop_duplicates(subset=[id_col], keep="first")
    if len(df) < before: print(f"[Clean] Dropped {before-len(df)} rows (missing/duplicate IDs).")
    df = df.sort_values(id_col).reset_index(drop=True)

    # ensure text_for_embedding
    if "text_for_embedding" not in df.columns:
        print("[Prep] 'text_for_embedding' missing; building from columns…")
        df["text_for_embedding"] = df.apply(build_text_for_embedding, axis=1)

    return df, id_col

def write_outputs(meta: pd.DataFrame, emb: np.ndarray, meta_out: str, npy_out: str, map_out: str, id_col: str):
    Path(os.path.dirname(meta_out)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(npy_out)).mkdir(parents=True, exist_ok=True)
    meta.to_csv(meta_out, index=False)
    np.save(npy_out, emb.astype(np.float32, copy=False))
    print(f"[IO] Wrote:\n  - {meta_out}\n  - {npy_out}  (shape={emb.shape})")

    map_df = pd.DataFrame({
        id_col: meta[id_col].values,
        "row_index": np.arange(len(meta), dtype=np.int32),
        "embedding_dim": emb.shape[1],
    })
    try:
        map_df.to_parquet(map_out, index=False)
        print(f"  - {map_out} (id→row map)")
    except Exception as e:
        fallback = os.path.splitext(map_out)[0] + ".csv"
        map_df.to_csv(fallback, index=False)
        print(f"[IO] Parquet failed ({e}); wrote CSV map: {fallback}")

# ---------- config / main ----------
def load_config():
    load_dotenv()
    pr = os.getenv("PROJECT_ROOT", os.getcwd())
    return {
        "IN_CSV":  os.getenv("COURSE_EMBED_META_CSV",  os.path.join(pr,"backend","data","interim","courses_metadata.csv")),
        "OUT_DIR": os.getenv("COURSE_EMB_OUT",         os.path.join(pr,"backend","data","embeddings")),
        "OUT_META":os.getenv("COURSE_EMBED_META_CLEAN",None),
        "OUT_NPY": os.getenv("COURSE_EMBED_NPY",       None),
        "OUT_MAP": os.getenv("COURSE_EMBED_MAP",       None),
        "MODEL":   os.getenv("COURSE_EMBED_MODEL", os.getenv("QUERY_EMBED_MODEL","sentence-transformers/all-MiniLM-L6-v2")),
        "BATCH":   int(os.getenv("COURSE_EMBED_BATCH","128")),
    }

def load_model(name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Model] {name} on {device}")
    return SentenceTransformer(name, device=device)

def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Embed course metadata to vectors.")
    parser.add_argument("--in_csv",   default=cfg["IN_CSV"])
    parser.add_argument("--out_meta", default=cfg["OUT_META"] or os.path.join(cfg["OUT_DIR"],"courses_metadata_clean.csv"))
    parser.add_argument("--out_npy",  default=cfg["OUT_NPY"]  or os.path.join(cfg["OUT_DIR"],"courses_embeddings.npy"))
    parser.add_argument("--out_map",  default=cfg["OUT_MAP"]  or os.path.join(cfg["OUT_DIR"],"course_embeddings_map.parquet"))
    parser.add_argument("--model",    default=cfg["MODEL"])
    parser.add_argument("--batch",    type=int, default=cfg["BATCH"])
    args = parser.parse_args()

    df, id_col = read_courses(args.in_csv)

    keep = [id_col,"title","description","platform","provider","level","url",
            "duration_hours","topic","skills","tags","text_for_embedding"]
    meta = df[[c for c in keep if c in df.columns]].copy()

    model = load_model(args.model)
    texts = meta["text_for_embedding"].fillna("").astype(str).tolist()
    print(f"[Embed] Encoding {len(texts)} courses…")
    emb = model.encode(texts, batch_size=args.batch, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)

    write_outputs(meta, emb, args.out_meta, args.out_npy, args.out_map, id_col)

if __name__ == "__main__":
    main()
