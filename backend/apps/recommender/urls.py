# backend/recommender/urls.py

from django.urls import path
from .views import UserFeaturesAPI

urlpatterns = [
    path('features/<int:student_id>/', UserFeaturesAPI.as_view(), name='user-features'),
]
