
from rest_framework.decorators import api_view
from rest_framework.response import Response
from apps.recommender.engine import get_recommendations

@api_view(["GET"])
def recommend_query(request):
    query = request.query_params.get("query", "").strip()
    level = request.query_params.get("level")
    try:
        top_k = int(request.query_params.get("top_k", 3))
    except (TypeError, ValueError):
        top_k = 3

    if not query:
        return Response({"error": "Missing required 'query' parameter"}, status=400)

    res = get_recommendations(query, level, top_k)
    best = res.get("best_type")

    # Choose primary list for "top"
    if best in ("books", "courses", "videos"):
        top_list = res.get(best, [])
    else:
        modal_lists = {k: v for k, v in res.items() if k in ("books", "courses", "videos")}
        top_key = max(modal_lists, key=lambda k: len(modal_lists[k])) if modal_lists else None
        top_list = modal_lists.get(top_key, []) if top_key else []

    supporting = {
        k: v for k, v in res.items()
        if k in ("books", "courses", "videos") and v is not top_list
    }

    payload = {
        "best_type": best,
        "top": top_list,
        "supporting": supporting,
        "surprise": res.get("surprise")  # <-- forward the engine's surprise
    }
    return Response(payload)
