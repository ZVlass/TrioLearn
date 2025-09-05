from django.contrib import admin
from django.db.models import Count
from .models_vectors import VectorizationRun, TopicVector
from .models import LearnerProfile, Course, Book, Video, Interaction

@admin.register(LearnerProfile)
class LearnerProfileAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'gender', 'region', 'highest_education',
        'age_band', 'course_prop', 'reading_prop', 'video_prop'
    )
    search_fields = ('user__username', 'region', 'highest_education')
    list_filter = ('gender', 'region', 'highest_education', 'age_band')

@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = (
        'title', 'platform', 'ratings', 'rating_level',
        'difficulty_num', 'students_enrolled', 'popularity'
    )
    search_fields = ('title', 'platform', 'rating_level')
    list_filter = ('platform', 'rating_level')

@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = (
        'title', 'authors', 'categories', 'language',
        'published_date', 'average_rating', 'ratings_count'
    )
    search_fields = ('title', 'authors', 'categories')
    list_filter = ('language', 'categories', 'published_date')

@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = (
        'title', 'channel', 'published_at', 'difficulty'
    )
    search_fields = ('title', 'channel')
    list_filter = ('channel', 'difficulty', 'published_at')

@admin.register(Interaction)
class InteractionAdmin(admin.ModelAdmin):
    list_display = (
        'learner', 'event_type', 'course', 'book', 'video', 'rating', 'timestamp'
    )
    search_fields = ('learner__user__username',)
    list_filter = ('event_type', 'timestamp')



# ---------- VectorizationRun ----------
@admin.register(VectorizationRun)
class VectorizationRunAdmin(admin.ModelAdmin):
    list_display = (
        "version_tag",
        "model_name",
        "vector_kind",
        "dim",
        "created_at",
        "vectors_count",
    )
    list_filter = ("vector_kind", "model_name", "dim", "created_at")
    search_fields = ("version_tag", "model_name")
    ordering = ("-created_at",)

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        # annotate how many TopicVectors belong to each run
        return qs.annotate(_vectors_count=Count("vectors"))

    @admin.display(description="Vectors", ordering="_vectors_count")
    def vectors_count(self, obj):
        return obj._vectors_count


# ---------- TopicVector ----------
@admin.register(TopicVector)
class TopicVectorAdmin(admin.ModelAdmin):
    # Keep list rows light—don’t render the whole vector
    list_display = (
        "modality",
        "external_id",
        "run",
        "model_name",
        "vector_kind",
        "dim",
        "created_at",
        "updated_at",
        "preview_dims",
    )
    list_filter = (
        "modality",
        "vector_kind",
        "model_name",
        ("run", admin.RelatedOnlyFieldListFilter),
        ("created_at", admin.DateFieldListFilter),
    )
    search_fields = (
        "external_id",
        "model_name",
        "run__version_tag",
    )
    autocomplete_fields = ("run",)
    list_select_related = ("run",)
    ordering = ("-created_at",)
    readonly_fields = ("dim", "created_at", "updated_at")

    @admin.display(description="vector[:5]")
    def preview_dims(self, obj):
        try:
            # obj.vector behaves like a Python list; show the first few elements
            head = list(obj.vector[:5])
            # compact string, e.g. [0.12, -0.03, ...]
            return "[" + ", ".join(f"{x:.2f}" for x in head) + "]"
        except Exception:
            return "—"

    
