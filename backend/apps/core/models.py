from django.db import models

# Create your models here.

from django.db import models
from django.contrib.auth.models import User

DIFFICULTY_LEVELS = [
    ('beginner', 'Beginner'),
    ('intermediate', 'Intermediate'),
    ('advanced', 'Advanced'),
]

class LearnerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    gender = models.CharField(max_length=10, blank=True)
    region = models.CharField(max_length=100, blank=True)
    highest_education = models.CharField(max_length=100, blank=True)
    age_band = models.CharField(max_length=20, blank=True)
    avg_session_duration_min = models.FloatField(null=True, blank=True)
    course_prop = models.FloatField(null=True, blank=True)
    reading_prop = models.FloatField(null=True, blank=True)
    video_prop = models.FloatField(null=True, blank=True)
    last_active = models.DateTimeField(auto_now=True)

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

    def __str__(self):
        return f"{self.learner.user.username} {self.event_type} at {self.timestamp}"
