from rest_framework.decorators import api_view
from rest_framework.response import Response
from apps.recommender.engine import get_recommendations
import numpy as np


@api_view(["GET"])
def recommend_query(request):
    query = request.query_params.get("query", "")
    level = request.query_params.get("level", "intermediate")
    top_k = int(request.query_params.get("top_k", 3))

    if not query:
        return Response({"error": "Missing required 'query' parameter"}, status=400)

    results = get_recommendations(query, level, top_k)

    def safe_df(df):
        return df.fillna("").replace([np.nan, np.inf, -np.inf], "").to_dict(orient="records")

    return Response({
        "best_type": results["best_type"],
        "top": safe_df(results["top"]),
        "supporting": {
            k: safe_df(v) for k, v in results["supporting"].items()
        }
    })
