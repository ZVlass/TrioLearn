from django.shortcuts import render

# Create your views here.

from rest_framework import viewsets
from .models import LearnerProfile, Course, Book, Video, Interaction
from .serializers import (
    LearnerProfileSerializer,
    CourseSerializer,
    BookSerializer,
    VideoSerializer,
    InteractionSerializer
)

class CourseViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Course.objects.all()
    serializer_class = CourseSerializer

class BookViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer

class VideoViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer

class LearnerProfileViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = LearnerProfile.objects.select_related('user')
    serializer_class = LearnerProfileSerializer

class InteractionViewSet(viewsets.ModelViewSet):
    queryset = Interaction.objects.all()
    serializer_class = InteractionSerializer

