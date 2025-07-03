from django.shortcuts import render

# Create your views here.
# backend/recommender/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import UserFeaturesSerializer
from features_loader import load_features_for_student  # your own utility

class UserFeaturesAPI(APIView):
    """Return the precomputed feature vector for a given student."""
    def get(self, request, student_id):
        feats = load_features_for_student(student_id)
        if not feats:
            return Response({"detail":"Not found"}, status=404)
        serializer = UserFeaturesSerializer(feats)
        return Response(serializer.data)
