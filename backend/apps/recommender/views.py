

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import UserFeaturesSerializer
from .user_features_loader import load_features_for_student

class UserFeaturesAPI(APIView):
    """
    GET /api/recommender/features/{student_id}/
    """
    def get(self, request, student_id: int):
        feats = load_features_for_student(student_id)
        if feats is None:
            return Response(
                {"detail": f"Student {student_id} not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        # Include the id_student in the payload
        feats["id_student"] = student_id
        serializer = UserFeaturesSerializer(feats)
        return Response(serializer.data)

