# backend/apps/core/management/commands/build_topics.py
from django.core.management.base import BaseCommand
from django.conf import settings

import pandas as pd
import numpy as np
from pathlib import Path
import re
import json
from typing import List, Optional
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
import joblib


def clean_text(s: Optional[str]) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def make_text(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        # fallback to any text-ish columns
        present = [c for c in df.columns if c.lower() in ("title", "description", "skills", "tags", "level")]
    return (
        df[present].astype(str)
        .agg(" ".join, axis=1)
        .map(clean_text)
    )


def load_items(path: Path, kind: str, id_candidates=("global_id", "course_id", "book_id", "video_id", "id")) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame(columns=["id", "kind", "text"])
    df = pd.read_csv(path)
    # choose a stable id column if present
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        df["id"] = np.arange(len(df))
    else:
        df = df.rename(columns={id_col: "id"})
    text = make_text(
        df,
        cols=["title", "short_description", "description", "skills", "tags", "level", "subtitle"]
    )
    return pd.DataFrame({"id": df["id"], "kind": kind, "text": text})


class Command(BaseCommand):
    help = "Train an LDA topic model over items and export per-item topic vectors + artifacts."

    def add_arguments(self, parser):
        base = Path(settings.BASE_DIR)

        parser.add_argument("--courses", type=Path,
            default=base / "data" / "intermediate" / "courses_combined_cleaned.csv",
            help="Path to courses CSV.")
        parser.add_argument("--books", type=Path,
            default=base / "data" / "intermediate" / "books_cleaned.csv",
            help="Path to books CSV (optional).")
        parser.add_argument("--videos", type=Path,
            default=base / "data" / "intermediate" / "videos_cleaned.csv",
            help="Path to videos CSV (optional).")
        parser.add_argument("--outdir", type=Path,
            default=base / "data" / "topics",
            help="Directory to write artifacts.")
        parser.add_argument("--k", type=int, default=30, help="Number of topics.")
        parser.add_argument("--max_iter", type=int, default=20, help="LDA max iterations.")
        parser.add_argument("--min_df", type=int, default=5, help="CountVectorizer min_df.")
        parser.add_argument("--max_df", type=float, default=0.5, help="CountVectorizer max_df.")
        parser.add_argument("--max_features", type=int, default=50000, help="Vocabulary cap.")
        parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
        parser.add_argument("--sample", type=int, default=0,
            help="If >0, sample this many rows uniformly for quick runs.")

    def handle(self, *args, **opts):
        outdir: Path = opts["outdir"]
        outdir.mkdir(parents=True, exist_ok=True)

        # 1) Load data
        frames = []
        frames.append(load_items(opts["courses"], "course"))
        frames.append(load_items(opts["books"], "book"))
        frames.append(load_items(opts["videos"], "video"))
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["text"]).drop_duplicates(subset=["id","kind"])

        if opts["sample"] and len(df) > opts["sample"]:
            df = df.sample(opts["sample"], random_state=opts["random_state"]).reset_index(drop=True)

        if df.empty:
            self.stderr.write(self.style.ERROR("No input rows found. Check your CSV paths."))
            return

        self.stdout.write(self.style.SUCCESS(f"[data] Loaded {len(df):,} items."))

        # 2) Vectorize
        vectorizer = CountVectorizer(
            stop_words=ENGLISH_STOP_WORDS,
            min_df=opts["min_df"],
            max_df=opts["max_df"],
            max_features=opts["max_features"]
        )
        X = vectorizer.fit_transform(df["text"].tolist())
        self.stdout.write(self.style.SUCCESS(f"[text] Vocab size: {len(vectorizer.vocabulary_):,}"))

        # 3) Train LDA
        lda = LatentDirichletAllocation(
            n_components=opts["k"],
            max_iter=opts["max_iter"],
            learning_method="batch",
            random_state=opts["random_state"],
            evaluate_every=0,
        )
        theta = lda.fit_transform(X)  # shape: (n_docs, K), rows sum to 1
        K = opts["k"]

        # 4) Save artifacts
        joblib.dump(vectorizer, outdir / "vectorizer.joblib")
        joblib.dump(lda, outdir / "lda.joblib")
        self.stdout.write(self.style.SUCCESS(f"[artifacts] Saved vectorizer & lda to {outdir}"))

        # 5) Per-item topic vectors → parquet (columns topic_0..topic_{K-1})
        topic_cols = [f"topic_{i:02d}" for i in range(K)]
        topics_df = pd.DataFrame(theta, columns=topic_cols)
        out_df = pd.concat([df[["id","kind"]].reset_index(drop=True), topics_df], axis=1)
        out_df.to_parquet(outdir / "item_topic_vectors.parquet", index=False)
        self.stdout.write(self.style.SUCCESS(f"[vectors] Wrote {len(out_df):,} rows to item_topic_vectors.parquet"))

        # 6) Human-readable top terms per topic
        def top_terms(component, vocab, n=12):
            inds = np.argsort(component)[::-1][:n]
            inv_vocab = {idx: term for term, idx in vocab.items()}
            return [inv_vocab[i] for i in inds if i in inv_vocab]

        rows = []
        for k in range(K):
            terms = top_terms(lda.components_[k], vectorizer.vocabulary_, n=12)
            rows.append({"topic": k, "top_terms": ", ".join(terms)})
        pd.DataFrame(rows).to_csv(outdir / "topics_top_terms.csv", index=False)
        self.stdout.write(self.style.SUCCESS(f"[report] topics_top_terms.csv written."))

        # 7) Minimal manifest for downstream code
        manifest = {
            "k": K,
            "vectorizer": str(outdir / "vectorizer.joblib"),
            "lda_model": str(outdir / "lda.joblib"),
            "vectors_parquet": str(outdir / "item_topic_vectors.parquet"),
            "top_terms_csv": str(outdir / "topics_top_terms.csv"),
        }
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.stdout.write(self.style.SUCCESS(f"[done] Manifest saved to {outdir/'manifest.json'}"))

