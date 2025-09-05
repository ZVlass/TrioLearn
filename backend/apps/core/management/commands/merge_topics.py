# backend/apps/core/management/commands/merge_topics.py
from django.core.management.base import BaseCommand
from django.conf import settings
from pathlib import Path
import pandas as pd
import os, re, ast

class Command(BaseCommand):
    help = "Merge book/course/video topic vectors into all_topic_vectors.parquet"

    def handle(self, *args, **opts):
        # 0) Resolve topics dir: ENV override, else BASE_DIR fallback; expand %VAR%/${VAR}/~
        default_dir = Path(settings.BASE_DIR) / "backend" / "data" / "topics"
        raw = os.getenv("TOPICS_DIR", str(default_dir))
        topics_dir = Path(os.path.expanduser(os.path.expandvars(raw))).resolve()
        self.stdout.write(f"[info] Using topics_dir: {topics_dir}")

        inputs = [
            ("book_topic_vectors.parquet",   "book"),
            ("course_topic_vectors.parquet", "course"),
            ("video_topic_vectors.parquet",  "video"),
        ]

        parts = []
        for fname, modality in inputs:
            p = topics_dir / fname
            if not p.exists():
                self.stderr.write(f"[skip] {p} not found")
                continue

            df = pd.read_parquet(p)  # pyarrow fast-path if installed
            df["modality"] = modality

            # --- external_id ---
            id_candidates = ["external_id","id","item_id","course_id","book_id","video_id","global_id"]
            ext_id = next((c for c in id_candidates if c in df.columns), None)
            if ext_id and ext_id != "external_id":
                df = df.rename(columns={ext_id: "external_id"})
            if "external_id" not in df.columns:
                df["external_id"] = [f"{modality}_{i}" for i in range(len(df))]

            # --- title ---
            title_candidates = ["title","name","course_title","book_title","video_title"]
            ttl = next((c for c in title_candidates if c in df.columns), None)
            if ttl and ttl != "title":
                df = df.rename(columns={ttl: "title"})
            if "title" not in df.columns:
                df["title"] = ""

            # --- topic_vector ---
            vec_candidates = ["topic_vector","topics","vector","topic_vec","lda_vector"]
            vcol = next((c for c in vec_candidates if c in df.columns), None)
            if vcol and vcol != "topic_vector":
                df = df.rename(columns={vcol: "topic_vector"})

            if "topic_vector" not in df.columns:
                # Try wide columns: topic_0, topic_1, ...
                topic_cols = [c for c in df.columns if re.fullmatch(r"topic_\d+", str(c))]
                if topic_cols:
                    df["topic_vector"] = df[topic_cols].apply(
                        lambda r: _to_float_list(r.values), axis=1
                    )
                else:
                    self.stderr.write(f"[warn] No topic vector found in {fname}. Available columns: {list(df.columns)}")
                    continue
            else:
                # Normalize whatever is inside topic_vector (lists, arrays, strings, Series)
                df["topic_vector"] = df["topic_vector"].apply(_coerce_vector_any)

            # Keep minimal schema per-row
            part = df[["modality","external_id","title","topic_vector"]].copy()
            parts.append(part)

        if not parts:
            self.stderr.write(f"No input topic parquet files with usable columns found in {topics_dir}")
            return

        # Concatenate and enforce clean dtypes
        out = pd.concat(parts, ignore_index=True)

        # Strong typing to avoid Arrow casting issues
        out["modality"]    = out["modality"].astype(str)
        out["external_id"] = out["external_id"].astype(str)
        out["title"]       = out["title"].fillna("").astype(str)
        out["topic_vector"] = out["topic_vector"].apply(_to_float_list)
        out["num_topics"]   = out["topic_vector"].apply(len)

        # Optional: drop rows with empty vectors
        # out = out[out["num_topics"] > 0].reset_index(drop=True)

        out_path = topics_dir / "all_topic_vectors.parquet"
        out.to_parquet(out_path, index=False)
        self.stdout.write(f"[OK] {out_path} (rows={len(out)})")


def _coerce_vector_any(v):
    """
    Accepts: list/tuple/np.array/Series or a string like "[0.1, 0.2]".
    Returns: clean list[float] (empty list on failure).
    """
    # If string looks like a list, try to parse
    if isinstance(v, str) and v.strip().startswith(("[", "(")):
        try:
            parsed = ast.literal_eval(v)
            return _to_float_list(parsed)
        except Exception:
            return []
    # If it's a Series/ndarray/other with tolist()
    if hasattr(v, "tolist"):
        try:
            return _to_float_list(v.tolist())
        except Exception:
            return []
    # If it's already list/tuple
    if isinstance(v, (list, tuple)):
        return _to_float_list(v)
    # Fallback
    return []


def _to_float_list(x):
    """
    Convert iterable x to list[float], dropping non-convertible entries.
    If x is scalar or None -> [].
    """
    try:
        # scalar guard
        if x is None or isinstance(x, (str, bytes)) or not hasattr(x, "__iter__"):
            return []
        out = []
        for val in x:
            try:
                out.append(float(val))
            except Exception:
                # skip non-numeric entries
                continue
        return out
    except Exception:
        return []
