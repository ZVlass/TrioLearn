
from django.db import models
from pgvector.django import VectorField, IvfflatIndex


# ---- constants to tune ----
VECTOR_DIM = 30
DIST_OPCLASS = "vector_cosine_ops"  


class VectorizationRun(models.Model):
    """
    Tracks a single vectorization build (for reproducibility & A/B testing).
    """
    version_tag = models.CharField(max_length=64, unique=True)  # e.g., "topics-2025-09-04"
    model_name = models.CharField(max_length=128)               # e.g., "LDA-50", "all-MiniLM-L6-v2"
    vector_kind = models.CharField(
        max_length=32,
        choices=[("topic_distribution", "topic_distribution"), ("embedding", "embedding")],
    )
    dim = models.PositiveIntegerField(default=VECTOR_DIM)

    parameters = models.JSONField(default=dict, blank=True)      # hyperparams, stopwords, etc.
    input_snapshot_hash = models.CharField(max_length=64, blank=True)  # optional integrity marker

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "vectorization_run"
        indexes = [
            models.Index(fields=["model_name"]),
            models.Index(fields=["vector_kind"]),
        ]

    def __str__(self):
        return f"{self.version_tag} · {self.model_name} · {self.vector_kind}"


class TopicVector(models.Model):
    """
    One row per item (book/course/video) per vectorization run.
    Uses (modality, external_id) to point back to the source item.
    """
    class Modality(models.TextChoices):
        BOOK = "book", "book"
        COURSE = "course", "course"
        VIDEO = "video", "video"

    # Link to the run that produced this vector
    run = models.ForeignKey(VectorizationRun, on_delete=models.CASCADE, related_name="vectors")

    # Identify the item this vector belongs to
    modality = models.CharField(max_length=16, choices=Modality.choices)
    external_id = models.CharField(max_length=128)  

    # Vector payload and metadata
    vector = VectorField(dimensions=VECTOR_DIM) 
    dim = models.PositiveIntegerField(default=VECTOR_DIM)
    model_name = models.CharField(max_length=128)  # copied for convenience/filtering
    vector_kind = models.CharField(
        max_length=32,
        choices=[("topic_distribution", "topic_distribution"), ("embedding", "embedding")],
    )
    meta = models.JSONField(default=dict, blank=True)  # provenance (e.g., file hash, script name)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "topic_vector"

        # Uniqueness: one vector per item per run (run.version_tag)
        constraints = [
            models.UniqueConstraint(
                fields=["run", "modality", "external_id"],
                name="uniq_item_per_run",
            ),
        ]

        indexes = [
            # Fast filtering
            models.Index(fields=["modality"]),
            models.Index(fields=["model_name"]),
            models.Index(fields=["vector_kind"]),

            # ANN index (IVFFLAT). Tune lists for your dataset size.
            IvfflatIndex(
                name="tv_vector_ivf_cos",
                fields=["vector"],
                opclasses=["vector_cosine_ops"],  # cosine distance
                lists=100,                 # start small
            ),
        ]

    def __str__(self):
        return f"{self.modality}:{self.external_id} · {self.run.version_tag}"
