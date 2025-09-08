
from django.db import models
from django.contrib.auth.models import User


DIFFICULTY_LEVELS = [
    ('beginner', 'Beginner'),
    ('intermediate', 'Intermediate'),
    ('advanced', 'Advanced'),
]

FORMAT_LABELS = {
    "video": "videos",
    "course": "interactive courses",
    "reading": "reading",
}

class LearnerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    gender = models.CharField(max_length=10, blank=True)
    region = models.CharField(max_length=100, blank=True)
    highest_education = models.CharField(max_length=100, blank=True)
    age_band = models.CharField(max_length=20, blank=True)
    avg_session_duration_min = models.FloatField(null=True, blank=True)

    topic_interests = models.JSONField(null=True, blank=True)
    preferred_format = models.CharField(max_length=20, blank=True)

    # Modality “signals” (can be learned or bootstrapped at registration)
    course_prop = models.FloatField(null=True, blank=True)
    reading_prop = models.FloatField(null=True, blank=True)
    video_prop = models.FloatField(null=True, blank=True)

    # Soft prior weight for preferred_format (0..1). Start ~0.3, decay with interactions.
    format_prior_weight = models.FloatField(null=True, blank=True, default=0.3)

    format_model_confidence = models.FloatField(null=True, blank=True)

    last_active = models.DateTimeField(auto_now=True)

    def matches_item(self, item_topics):
        if not item_topics:
            return False

        # Cold-start: if the user hasn't picked topics yet, allow everything.
        if not self.topic_interests:
            return True

        top_tags = sorted(item_topics.items(), key=lambda x: x[1], reverse=True)[:3]
        return any(tag for tag, _ in top_tags if tag in (self.topic_interests or []))

    # --- Recommendation helpers (used by dashboard banner) ---

    def _signal_distribution(self):
        """Return normalized distribution from stored modality props."""
        props = {
            "course": self.course_prop or 0.0,
            "reading": self.reading_prop or 0.0,
            "video": self.video_prop or 0.0,
        }
        s = sum(props.values()) or 1.0
        return {k: v / s for k, v in props.items()}

    def blended_format_distribution(self):
        """
        Blend user preferred_format as a soft prior with learned props.
        If preferred is 'none' or empty, alpha=0.
        """
        p_model = self._signal_distribution()
        preferred = (self.preferred_format or "none").strip().lower()
        alpha = self.format_prior_weight or 0.0
        if preferred == "none":
            alpha = 0.0

        # user prior
        p_user = {"video": 0.0, "course": 0.0, "reading": 0.0}
        if preferred in p_user:
            p_user[preferred] = 1.0

        p = {k: alpha * p_user[k] + (1 - alpha) * p_model.get(k, 0.0) for k in p_user}
        s = sum(p.values()) or 1.0
        return {k: v / s for k, v in p.items()}

    def top_format_key_and_label(self):
        p = self.blended_format_distribution()
        top_key = max(p, key=p.get)
        return top_key, FORMAT_LABELS.get(top_key, top_key)

    def __str__(self):
        return self.user.username

class Course(models.Model):
    title = models.CharField(max_length=300)
    ratings = models.FloatField(null=True, blank=True)
    rating_level = models.CharField(max_length=20, blank=True, null=True)
    difficulty_num = models.FloatField(null=True, blank=True)
    students_enrolled = models.IntegerField(null=True, blank=True)
    popularity = models.FloatField(null=True, blank=True)
    platform = models.CharField(max_length=100, default="Coursera")
    description = models.TextField(blank=True)
    url = models.URLField(blank=True)
    topic_vector = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class Book(models.Model):
    keyword = models.CharField(max_length=100, blank=True)
    title = models.CharField(max_length=300)
    authors = models.CharField(max_length=300)
    description = models.TextField()
    categories = models.CharField(max_length=200, blank=True)
    published_date = models.CharField(max_length=20, blank=True)
    page_count = models.IntegerField(null=True, blank=True)
    language = models.CharField(max_length=10, blank=True)
    average_rating = models.FloatField(null=True, blank=True)
    ratings_count = models.IntegerField(null=True, blank=True)
    preview_link = models.URLField(blank=True)
    info_link = models.URLField(blank=True)
    topic_vector = models.JSONField(blank=True, null=True)

    gbooks_id = models.CharField(max_length=64, unique=True, null=True, blank=True)
    isbn_13 = models.CharField(max_length=13, unique=True, null=True, blank=True)
    thumbnail = models.URLField(blank=True)

    def __str__(self):
        return self.title


class Video(models.Model):
    video_id = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=300)
    description = models.TextField()
    channel = models.CharField(max_length=200)
    published_at = models.DateTimeField()
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_LEVELS, blank=True)
    topic_vector = models.JSONField(blank=True, null=True)

    @property
    def url(self):
        return f"https://youtu.be/{self.video_id}"

    def __str__(self):
        return self.title


class Interaction(models.Model):
    learner = models.ForeignKey(LearnerProfile, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.SET_NULL, null=True, blank=True)
    book = models.ForeignKey(Book, on_delete=models.SET_NULL, null=True, blank=True)
    video = models.ForeignKey(Video, on_delete=models.SET_NULL, null=True, blank=True)

    # Behavior tracking
    event_type = models.CharField(max_length=50, choices=[
        ('click', 'Click'),
        ('view', 'View'),
        ('like', 'Like'),
        ('bookmark', 'Bookmark'),
        ('rating', 'Rating'),
        ('complete', 'Complete')
    ])
    rating = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    @property
    def item(self):
        return self.course or self.book or self.video

    def __str__(self):
        return f"{self.learner.user.username} {self.event_type} at {self.timestamp}"