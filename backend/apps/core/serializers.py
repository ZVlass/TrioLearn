
from rest_framework import serializers
from .models import LearnerProfile, Course, Book, Video, Interaction
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

class LearnerProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer()

    class Meta:
        model = LearnerProfile
        fields = [
            'id', 'user', 'gender', 'region', 'highest_education',
            'imd_band', 'age_band', 'avg_session_duration_min',
            'course_prop', 'reading_prop', 'video_prop', 'last_active'
        ]

class CourseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Course
        fields = [
            'id', 'title', 'ratings', 'rating_level', 'difficulty_num',
            'students_enrolled', 'popularity', 'platform', 'description',
            'url', 'topic_vector', 'created_at'
        ]

class BookSerializer(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = [
            'id', 'keyword', 'title', 'authors', 'description', 'categories',
            'published_date', 'page_count', 'language', 'average_rating',
            'ratings_count', 'preview_link', 'info_link', 'topic_vector'
        ]

class VideoSerializer(serializers.ModelSerializer):
    url = serializers.SerializerMethodField()

    class Meta:
        model = Video
        fields = [
            'id', 'video_id', 'title', 'description', 'channel',
            'published_at', 'difficulty', 'topic_vector', 'url'
        ]

    def get_url(self, obj):
        return obj.url

class InteractionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Interaction
        fields = [
            'id', 'learner', 'course', 'book', 'video',
            'rating', 'liked', 'event_type', 'timestamp'
        ]




