from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views_tracking import track_interaction
from .views import (
    CourseViewSet, BookViewSet, VideoViewSet,
    LearnerProfileViewSet, InteractionViewSet,
    landing, register_user, login_user, logout_user, dashboard
)

router = DefaultRouter()
router.register(r'courses', CourseViewSet)
router.register(r'books', BookViewSet)
router.register(r'videos', VideoViewSet)
router.register(r'learners', LearnerProfileViewSet)
router.register(r'interactions', InteractionViewSet)

urlpatterns = [
    # Web pages
    path('', landing, name='landing'),
    path('register/', register_user, name='register'),
    path('login/', login_user, name='login'),
    path('logout/', logout_user, name='logout'),
    path('dashboard/', dashboard, name='dashboard'),

    # API (prefixed to avoid root conflicts)
    path('api/', include(router.urls)),

    # Tracking
    path('track/', track_interaction, name='track_interaction'),
]
