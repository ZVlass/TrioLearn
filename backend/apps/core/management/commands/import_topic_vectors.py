# backend/apps/core/management/commands/import_topic_vectors.py
import ast
import json
import math
import os
import re
from typing import Iterable, List, Optional

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.core.models_vectors import (
    TopicVector,
    VectorizationRun,
    VECTOR_DIM, 
)

EXT_ID_CANDIDATES = [
    "external_id", "item_id", "global_id", "id",
    "isbn_13", "gbooks_id", "course_id", "video_id", "book_id",
]

class Command(BaseCommand):
    help = "Import topic vectors from a Parquet file into TopicVector (pgvector)."

    def add_arguments(self, parser):
        load_dotenv()  

        parser.add_argument("--file",
            default=os.getenv("TOPIC_VECTORS_FILE", os.path.join(os.getenv("TOPICS_DIR",""), "all_topic_vectors.parquet")))
        parser.add_argument("--run-tag",
            default=os.getenv("TOPIC_RUN_TAG", "topics-v1"))
        parser.add_argument("--model-name",
            default=os.getenv("TOPIC_MODEL_NAME"))
        parser.add_argument("--vector-kind",
            choices=["topic_distribution","embedding"],
            default=os.getenv("TOPIC_VECTOR_KIND"))
        parser.add_argument("--modality-col", default="modality")
        parser.add_argument("--external-id-col", default=None)
        parser.add_argument("--vector-col", default=None)
        parser.add_argument("--dim",
            type=int, default=int(os.getenv("TOPIC_DIM", "50")))
        parser.add_argument("--batch-size",
            type=int, default=int(os.getenv("TOPIC_BATCH_SIZE", "1000")))
        parser.add_argument("--replace", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--limit", type=int, default=None)

    def handle(self, *args, **opts):
        path = opts["file"]
        run_tag = opts["run_tag"]
        modality_col = opts["modality_col"]
        ext_col = opts["external_id_col"]
        vec_col = opts["vector_col"]
        exp_dim = opts["dim"]
        batch_size = max(1, int(opts["batch_size"]))
        replace = bool(opts["replace"])
        dry_run = bool(opts["dry_run"])
        limit = opts["limit"]
        override_model_name = opts["model_name"]
        override_vector_kind = opts["vector_kind"]

        self.stdout.write(f"[io] reading parquet: {path}")
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            raise CommandError(f"Failed to read parquet: {e}")

        if limit:
            df = df.head(limit)

        if modality_col not in df.columns:
            raise CommandError(f"Missing modality column '{modality_col}'. Columns: {df.columns.tolist()}")

        # external_id detection
        if not ext_col:
            ext_col = next((c for c in EXT_ID_CANDIDATES if c in df.columns), None)
            if not ext_col:
                raise CommandError(
                    f"Could not detect an external id column. "
                    f"Tried {EXT_ID_CANDIDATES}. Columns: {df.columns.tolist()}"
                )
        elif ext_col not in df.columns:
            raise CommandError(f"External id column '{ext_col}' not found in parquet.")

        # vector detection
        vectors: List[List[float]]
        topic_cols: Optional[List[str]] = None
        if vec_col:
            if vec_col not in df.columns:
                raise CommandError(f"Vector column '{vec_col}' not found.")
            vectors = [self._coerce_vec(v) for v in df[vec_col].tolist()]
        else:
            # detect topic_#
            topic_cols = [c for c in df.columns if re.fullmatch(r"topic_\d+", c)]
            if topic_cols:
                topic_cols = sorted(topic_cols, key=lambda x: int(x.split("_")[1]))
                vectors = df[topic_cols].astype(float).values.tolist()
            elif "vector" in df.columns:
                vectors = [self._coerce_vec(v) for v in df["vector"].tolist()]
            else:
                raise CommandError("No vector column found and no topic_* columns present.")

        if not vectors:
            self.stdout.write(self.style.WARNING("[warn] No vectors found; nothing to import."))
            return

        # dimension checks
        inferred_dim = len(vectors[0])
        if exp_dim is not None and inferred_dim != exp_dim:
            raise CommandError(f"Data dim {inferred_dim} != --dim {exp_dim}")
        if inferred_dim != VECTOR_DIM:
            raise CommandError(
                f"Data dim {inferred_dim} != model VECTOR_DIM {VECTOR_DIM}. "
                f"Either change models_vectors.VECTOR_DIM and re-migrate, or export {VECTOR_DIM}-dim vectors."
            )

        # create or fetch run
        run, created = VectorizationRun.objects.get_or_create(
            version_tag=run_tag,
            defaults=dict(
                model_name=override_model_name or "LDA-50",
                vector_kind=override_vector_kind or "topic_distribution",
                dim=inferred_dim,
            ),
        )
        if not created:
            # update run metadata if overrides provided
            dirty = False
            if override_model_name and run.model_name != override_model_name:
                run.model_name = override_model_name; dirty = True
            if override_vector_kind and run.vector_kind != override_vector_kind:
                run.vector_kind = override_vector_kind; dirty = True
            if run.dim != inferred_dim:
                run.dim = inferred_dim; dirty = True
            if dirty:
                run.save(update_fields=["model_name", "vector_kind", "dim"])

        if replace and not dry_run:
            self.stdout.write(f"[ops] deleting existing vectors for run '{run.version_tag}' …")
            TopicVector.objects.filter(run=run).delete()

        # build objects
        rows = df[[modality_col, ext_col]].itertuples(index=False, name=None)
        to_insert: List[TopicVector] = []
        for (modality, ext_id), vec in zip(rows, vectors):
            to_insert.append(TopicVector(
                run=run,
                modality=str(modality),
                external_id=str(ext_id),
                vector=[float(x) for x in vec],
                dim=inferred_dim,
                model_name=run.model_name,
                vector_kind=run.vector_kind,
            ))

        self.stdout.write(f"[plan] rows: {len(to_insert)}, dim: {inferred_dim}, run: {run.version_tag}")
        if dry_run:
            self.stdout.write(self.style.SUCCESS("[dry-run] parsed OK, no inserts performed."))
            return

        # bulk insert in batches
        total = 0
        with transaction.atomic():
            for i in range(0, len(to_insert), batch_size):
                chunk = to_insert[i:i + batch_size]
                TopicVector.objects.bulk_create(chunk, batch_size=batch_size, ignore_conflicts=True)
                total += len(chunk)
                if len(to_insert) > batch_size:
                    self.stdout.write(f"[insert] {min(total, len(to_insert))}/{len(to_insert)}")

        count = TopicVector.objects.filter(run=run).count()
        self.stdout.write(self.style.SUCCESS(f"[done] inserted (or already present): {total}; now stored for run: {count}"))

    # --- helpers ---
    def _coerce_vec(self, v) -> List[float]:
        """
        Accept list/tuple/np.array; if string like '[0.1, 0.2]', parse; if JSON, parse too.
        """
        if isinstance(v, (list, tuple, np.ndarray)):
            return [float(x) for x in v]
        if isinstance(v, str):
            s = v.strip()
            try:
                # try JSON first
                if s.startswith("[") and s.endswith("]"):
                    parsed = json.loads(s)
                    return [float(x) for x in parsed]
                # try Python literal
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [float(x) for x in parsed]
            except Exception:
                pass
        # last resort: iterable
        try:
            return [float(x) for x in v]
        except Exception:
            raise CommandError(f"Cannot coerce vector value: {v!r}")
