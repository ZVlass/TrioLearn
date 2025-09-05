
from django.core.management.base import BaseCommand
import os
import re
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
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
        present = [c for c in df.columns if c.lower() in ("title", "description", "skills", "tags", "level", "subtitle")]
    return df[present].astype(str).agg(" ".join, axis=1).map(clean_text)


def load_items(path: Path, kind: str, id_candidates=("global_id", "course_id", "book_id", "video_id", "id")) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{kind} CSV not found: {path}")
    df = pd.read_csv(path)
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        df["id"] = np.arange(len(df))
    else:
        df = df.rename(columns={id_col: "id"})
    text = make_text(df, ["title", "short_description", "description", "skills", "tags", "level", "subtitle"])
    return pd.DataFrame({"id": df["id"], "kind": kind, "text": text})


class Command(BaseCommand):
    help = "Train an LDA topic model using ONLY paths & hyperparams from .env"

    def handle(self, *args, **opts):
        # 1) Load the nearest .env and remember its folder for resolving relative paths
        env_path = Path(find_dotenv(usecwd=True))
        if not env_path:
            raise RuntimeError("No .env found. Place one in your project and try again.")
        load_dotenv(env_path)
        env_base = env_path.parent  # resolve relative paths against the .env location

        def need(key: str) -> str:
            v = os.getenv(key)
            if not v:
                raise RuntimeError(f"Missing required env var: {key}")
            return v

        def need_path(key: str) -> Path:
            raw = need(key)
            p = Path(raw)
            return p if p.is_absolute() else (env_base / p)

        # 2) Paths (env-only)
        courses = need_path("COURSES_CSV")
        books   = need_path("BOOKS_CSV")
        videos  = need_path("VIDEOS_CSV")
        outdir  = need_path("TOPICS_OUTDIR")
        outdir.mkdir(parents=True, exist_ok=True)

        # 3) Hyperparams (env-only, with sensible defaults)
        k            = int(os.getenv("TOPICS_K", 30))
        max_iter     = int(os.getenv("TOPICS_MAX_ITER", 20))
        min_df       = int(os.getenv("TOPICS_MIN_DF", 5))
        max_df       = float(os.getenv("TOPICS_MAX_DF", 0.5))
        max_features = int(os.getenv("TOPICS_MAX_FEATURES", 50_000))
        random_state = int(os.getenv("TOPICS_RANDOM_STATE", 42))
        sample       = int(os.getenv("TOPICS_SAMPLE", 0))

        self.stdout.write(self.style.SUCCESS("[config] Loaded from .env only"))
        for kname, p in [("courses", courses), ("books", books), ("videos", videos), ("outdir", outdir)]:
            self.stdout.write(self.style.SUCCESS(f"[config] {kname} = {p.resolve()}"))
        self.stdout.write(self.style.SUCCESS(
            f"[hp] k={k} max_iter={max_iter} min_df={min_df} max_df={max_df} "
            f"max_features={max_features} random_state={random_state} sample={sample}"
        ))

        # 4) Load data
        frames = [
            load_items(courses, "course"),
            load_items(books, "book"),
            load_items(videos, "video"),
        ]
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["text"]).drop_duplicates(subset=["id", "kind"])
        if sample and len(df) > sample:
            df = df.sample(sample, random_state=random_state).reset_index(drop=True)
        if df.empty:
            raise RuntimeError("No input rows found after loading CSVs and cleaning.")

        self.stdout.write(self.style.SUCCESS(f"[data] Loaded {len(df):,} items."))

        # 5) Text → BoW
        vectorizer = CountVectorizer(
            stop_words=ENGLISH_STOP_WORDS,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )
        X = vectorizer.fit_transform(df["text"].tolist())
        self.stdout.write(self.style.SUCCESS(f"[text] Vocab size: {len(vectorizer.vocabulary_):,}"))

        # 6) LDA
        lda = LatentDirichletAllocation(
            n_components=k,
            max_iter=max_iter,
            learning_method="batch",
            random_state=random_state,
            evaluate_every=0,
        )
        theta = lda.fit_transform(X)

        # 7) Save artifacts
        joblib.dump(vectorizer, outdir / "vectorizer.joblib")
        joblib.dump(lda, outdir / "lda.joblib")
        self.stdout.write(self.style.SUCCESS(f"[artifacts] Saved vectorizer & lda → {outdir}"))

        # 8) Topic vectors
        topic_cols = [f"topic_{i:02d}" for i in range(k)]
        topics_df = pd.DataFrame(theta, columns=topic_cols)
        out_df = pd.concat([df[["id", "kind"]].reset_index(drop=True), topics_df], axis=1)
        try:
            out_df.to_parquet(outdir / "item_topic_vectors.parquet", index=False)
            vectors_path = str(outdir / "item_topic_vectors.parquet")
        except Exception as e:
            self.stderr.write(self.style.WARNING(f"[warn] Parquet failed ({e}); writing CSV instead."))
            out_df.to_csv(outdir / "item_topic_vectors.csv", index=False)
            vectors_path = str(outdir / "item_topic_vectors.csv")
        self.stdout.write(self.style.SUCCESS(f"[vectors] Wrote {len(out_df):,} rows"))

        # 9) Human-readable topic terms
        inv_vocab = {idx: term for term, idx in vectorizer.vocabulary_.items()}
        rows = []
        for t, comp in enumerate(lda.components_):
            inds = np.argsort(comp)[::-1][:12]
            rows.append({"topic": t, "top_terms": ", ".join(inv_vocab.get(i, "") for i in inds)})
        pd.DataFrame(rows).to_csv(outdir / "topics_top_terms.csv", index=False)
        self.stdout.write(self.style.SUCCESS("[report] topics_top_terms.csv written."))

        # 10) Manifest
        manifest = {
            "k": k,
            "vectorizer": str(outdir / "vectorizer.joblib"),
            "lda_model": str(outdir / "lda.joblib"),
            "vectors": vectors_path,
            "top_terms_csv": str(outdir / "topics_top_terms.csv"),
        }
        (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.stdout.write(self.style.SUCCESS(f"[done] Manifest saved to {outdir/'manifest.json'}"))
