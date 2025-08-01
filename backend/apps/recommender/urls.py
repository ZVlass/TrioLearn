from django.urls import path
from .views import recommend_query

urlpatterns = [
    path("recommend_query/", recommend_query),
]

