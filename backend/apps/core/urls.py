from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    CourseViewSet, BookViewSet, VideoViewSet,
    LearnerProfileViewSet, InteractionViewSet
)
from . import views

router = DefaultRouter()
router.register(r'courses', CourseViewSet)
router.register(r'books', BookViewSet)
router.register(r'videos', VideoViewSet)
router.register(r'learners', LearnerProfileViewSet)
router.register(r'interactions', InteractionViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('register/', views.register_user, name='register'),
    path('login/', views.login_user, name='login'), 
    path('logout/', views.logout_user, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
]
