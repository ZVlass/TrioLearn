
import json
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from .models import Course, Book, Video, Interaction, LearnerProfile
from apps.recommender.utils import update_modality_signal

@require_POST
@login_required
def track_interaction(request):
    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        return HttpResponseBadRequest("Invalid JSON")

    item_type = data.get("item_type")   # 'course' | 'book' | 'video'
    item_id   = data.get("item_id")
    event     = data.get("event", "click")

    if item_type not in {"course", "book", "video"} or not item_id:
        return HttpResponseBadRequest("Bad payload")

    profile = LearnerProfile.objects.get(user=request.user)

    obj, modality = None, None
    if item_type == "course":
        obj, modality = Course.objects.filter(id=item_id).first(), "course"
    elif item_type == "book":
        obj, modality = Book.objects.filter(id=item_id).first(), "reading"
    elif item_type == "video":
        obj, modality = Video.objects.filter(id=item_id).first(), "video"

    if not obj:
        return HttpResponseBadRequest("Item not found")

    Interaction.objects.create(
        learner=profile,
        course=obj if item_type == "course" else None,
        book=obj if item_type == "book" else None,
        video=obj if item_type == "video" else None,
        event_type=event,
    )

    update_modality_signal(profile, modality, strength=1.0, decay=0.9)
    return JsonResponse({"ok": True})
