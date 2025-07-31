from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    CourseViewSet, BookViewSet, VideoViewSet,
    LearnerProfileViewSet, InteractionViewSet
)

router = DefaultRouter()
router.register(r'courses', CourseViewSet)
router.register(r'books', BookViewSet)
router.register(r'videos', VideoViewSet)
router.register(r'learners', LearnerProfileViewSet)
router.register(r'interactions', InteractionViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
