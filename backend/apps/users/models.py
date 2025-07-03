

from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.postgres.fields import ArrayField
from django.db.models import JSONField


class CustomUser(AbstractUser):
    """
    Extends Django's built-in AbstractUser to include learning-related fields.
    """
    preferred_modalities = ArrayField(
        base_field=models.CharField(max_length=32),
        default=list,
        help_text="Order of modalities the user prefers, e.g. ['video','text','interactive']"
    )

    knowledge_state = JSONField(
        default=dict,
        help_text="Per-topic skill levels, values between 0 and 1"
    )

    cold_start_cluster = models.CharField(
        max_length=64,
        blank=True,
        null=True,
        help_text="Cluster id for cold-start assignment"
    )

    profile_updated = models.DateTimeField(
        auto_now=True,
        help_text="Timestamp of last profile update (for decaying old data)"
    )

    def __str__(self):
        return f"{self.username} ({self.email})"


class UserProfile(models.Model):
    """
    Additional profile info for display or demographic filtering.
    """
    user = models.OneToOneField(
        "users.CustomUser",
        on_delete=models.CASCADE,
        related_name="profile"
    )
    bio = models.TextField(blank=True)
    location = models.CharField(max_length=128, blank=True)
    interests = models.TextField(blank=True)

    def __str__(self):
        return f"Profile for {self.user.username}"


class Course(models.Model):
    """
    Simplified course model for relations.
    """
    title = models.CharField(max_length=255)
    platform = models.CharField(max_length=100)
    level = models.CharField(
        max_length=32,
        choices=(
            ("beginner", "Beginner"),
            ("intermediate", "Intermediate"),
            ("advanced", "Advanced"),
        )
    )

    def __str__(self):
        return self.title


class UserInteraction(models.Model):
    """
    Track implicit feedback: views, clicks, completions, etc.
    """
    ACTION_CHOICES = (
        ("view", "Viewed"),
        ("click", "Clicked"),
        ("complete", "Completed"),
        ("like", "Liked"),
    )

    user = models.ForeignKey(
        "users.CustomUser",
        on_delete=models.CASCADE,
        related_name="interactions"
    )
    course = models.ForeignKey(
        Course,
        on_delete=models.CASCADE,
        related_name="interactions"
    )
    action = models.CharField(max_length=16, choices=ACTION_CHOICES)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.user.username} {self.action} {self.course.title} @ {self.timestamp}"
