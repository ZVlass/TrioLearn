
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm, TopicSelectionForm  
from apps.recommender.engine import get_recommendations


from apps.core.constants import TOPIC_LABELS


from rest_framework import viewsets
from .models import LearnerProfile, Course, Book, Video, Interaction
from .serializers import (
    LearnerProfileSerializer,
    CourseSerializer,
    BookSerializer,
    VideoSerializer,
    InteractionSerializer
)

class CourseViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Course.objects.all()
    serializer_class = CourseSerializer

class BookViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializer

class VideoViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer

class LearnerProfileViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = LearnerProfile.objects.select_related('user')
    serializer_class = LearnerProfileSerializer

class InteractionViewSet(viewsets.ModelViewSet):
    queryset = Interaction.objects.all()
    serializer_class = InteractionSerializer

def register_user(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data  

            # Create the user
            user = User.objects.create_user(
                username=data['username'],
                email=data['email'],
                password=data['password'],
            )
            # Save first/last name (optional fields)
            user.first_name = (data.get('first_name') or '').strip()
            user.last_name  = (data.get('last_name') or '').strip()
            user.save()

            # Preferred format (may be empty/None)
            preferred = (data.get('preferred_format') or 'none').strip().lower()

            # Soft-prior init for modality props
            if preferred in ('course', 'reading', 'video'):
                course_prop  = 0.6 if preferred == 'course'  else 0.2
                reading_prop = 0.6 if preferred == 'reading' else 0.2
                video_prop   = 0.6 if preferred == 'video'   else 0.2
                prior_weight = 0.3   # let model blend/decay this
            else:
                # "No preference": keep neutral start and no prior bias
                course_prop = reading_prop = video_prop = 0.2
                prior_weight = 0.0

            # Create profile
            profile = LearnerProfile.objects.create(
                user=user,
                gender=data['gender'],
                region=data['region'],
                highest_education=data['highest_education'],
                age_band=data['age_band'],
                topic_interests=data.get('topic_interests', []),
                preferred_format=preferred,
                course_prop=course_prop,
                reading_prop=reading_prop,
                video_prop=video_prop,
                format_prior_weight=prior_weight,  # if you added this field
            )

            login(request, user)
            messages.success(request, "Welcome to TrioLearn!")
            return redirect('dashboard')
    else:
        form = UserRegistrationForm()

    return render(request, 'core/register.html', {'form': form})


def login_user(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('dashboard')  # Update this to your actual home/dashboard view name
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'core/login.html')


def logout_user(request):
    logout(request)
    messages.success(request, "You have successfully logged out.")
    return redirect('login')  # Redirect to login page after logout

# ---------- helpers ----------

def _as_item_dict(obj):
    """
    Normalize a recommendation into the dict shape the template expects.
    Works for either model instances (Course/Book/Video) or pre-made dicts.
    """
    # If engine already returns dicts, try to map gracefuly
    if isinstance(obj, dict):
        href = obj.get("href") or obj.get("url") or obj.get("info_link") or "#"
        title = obj.get("display_title") or obj.get("title") or "Untitled"
        disp_id = obj.get("display_id") or obj.get("external_id") or obj.get("id") or ""
        return {
            "href": href,
            "display_title": title,
            "display_id": disp_id,
            "platform": obj.get("platform"),
            "provider": obj.get("provider"),
        }

    # Otherwise, convert our ORM objects
    if isinstance(obj, Course):
        return {
            "href": getattr(obj, "url", "#"),
            "display_title": getattr(obj, "title", "Untitled course"),
            "display_id": getattr(obj, "external_id", getattr(obj, "id", "")),
            "platform": getattr(obj, "platform", None),
            "provider": getattr(obj, "provider", None),
        }
    if isinstance(obj, Book):
        return {
            "href": getattr(obj, "info_link", "#"),
            "display_title": getattr(obj, "title", "Untitled book"),
            "display_id": getattr(obj, "external_id", getattr(obj, "id", "")),
            "platform": getattr(obj, "platform", None),
            "provider": getattr(obj, "provider", None),
        }
    if isinstance(obj, Video):
        return {
            "href": getattr(obj, "url", "#"),
            "display_title": getattr(obj, "title", "Untitled video"),
            "display_id": getattr(obj, "external_id", getattr(obj, "id", "")),
            "platform": getattr(obj, "platform", None),
            "provider": getattr(obj, "provider", None),
        }
    # Fallback
    return {"href": "#", "display_title": str(obj)[:80], "display_id": ""}

def _two(xs):
    return list(xs or [])[:2]

def _plural_best_type(top_key):
    """Map old single-key style to new plural used by the template."""
    m = {"course": "courses", "reading": "books", "video": "videos"}
    return m.get((top_key or "").lower(), "courses")

# ---------- main view ----------

@login_required
def dashboard(request):
    profile = LearnerProfile.objects.get(user=request.user)

    # --- topics (show past choices & read current focus) ---
    focus_topic = (request.GET.get("topic") or "").strip() or None
    if focus_topic and profile.topic_interests and focus_topic not in profile.topic_interests:
        # Ignore invalid topic code
        focus_topic = None

    topic_display = [(c, TOPIC_LABELS.get(c, c)) for c in (profile.topic_interests or [])]
    focus_label = TOPIC_LABELS.get(focus_topic) if focus_topic else None

    # --- call your engine: get_recommendations ---
    # Adjust signature if yours differs. Common patterns:
    #   get_recommendations(user=profile.user, topic=focus_topic, k=12)
    #   get_recommendations(profile, focus_topic=..., k=12)
    #   get_recommendations(query=?, level=?, ...)
    recs = get_recommendations(
        user=profile.user,
        topic=focus_topic,   # None means "all topics"
        k=12,                # we’ll cut to 2 in the UI
    )

    # Expect keys like 'courses'/'books'/'videos' (lists) and optional 'best_type' or 'top_key'
    courses_raw = recs.get("courses", [])
    books_raw   = recs.get("books", [])
    videos_raw  = recs.get("videos", [])

    # If your engine returns a best type, use it; otherwise fall back to profile’s top key
    best_type = recs.get("best_type")
    if not best_type:
        # Some older paths use 'top_key' as 'course'|'reading'|'video'
        top_key = recs.get("top_key")
        if not top_key and hasattr(profile, "top_format_key_and_label"):
            top_key, _ = profile.top_format_key_and_label()
        best_type = _plural_best_type(top_key)

    # Normalize items and **limit to 2** per modality
    formatted = {
        "courses": _two([_as_item_dict(x) for x in courses_raw]),
        "books":   _two([_as_item_dict(x) for x in books_raw]),
        "videos":  _two([_as_item_dict(x) for x in videos_raw]),
    }

    top = formatted.get(best_type, [])
    supporting = {
        "courses": formatted["courses"] if best_type != "courses" else [],
        "books":   formatted["books"]   if best_type != "books"   else [],
        "videos":  formatted["videos"]  if best_type != "videos"  else [],
    }

    # Surprise pick (optional)
    s_obj = recs.get("surprise")
    surprise = _as_item_dict(s_obj) if s_obj else None
    if surprise and "modality" not in surprise:
        # try to infer a modality tag for analytics (optional)
        surprise["modality"] = (
            "course" if surprise in formatted["courses"]
            else "book" if surprise in formatted["books"]
            else "video" if surprise in formatted["videos"]
            else recs.get("surprise_modality", "unknown")
        )
    # Optional reason
    if surprise and "surprise_reason" not in surprise:
        surprise["surprise_reason"] = recs.get("surprise_reason")

    # Banner labels (unchanged from your previous view)
    FORMAT_LABELS = {
        "video": "Videos",
        "course": "Courses",
        "reading": "Books & Articles",
        "none": "No preference",
    }
    preferred_format_label = FORMAT_LABELS.get((profile.preferred_format or "none"))

    # If you want to show what the model currently thinks is best:
    if hasattr(profile, "top_format_key_and_label"):
        _, top_format_label = profile.top_format_key_and_label()
    else:
        # derive a friendly label from best_type
        best_label_map = {"courses": "Courses", "books": "Books & Articles", "videos": "Videos"}
        top_format_label = best_label_map.get(best_type, "Courses")

    # Gender label (unchanged)
    GENDER_LABELS = {"M": "Male", "F": "Female", "O": "Other",
                     "Male": "Male", "Female": "Female", "Other": "Other"}
    gender_label = GENDER_LABELS.get(profile.gender, profile.gender or "")

    return render(request, "core/dashboard.html", {
        # profile + user info
        "profile": profile,
        "gender_label": gender_label,

        # topics (badges + dropdown)
        "topic_display": topic_display,
        "focus_topic": focus_topic,
        "focus_label": focus_label,

        # banner labels
        "preferred_format_label": preferred_format_label,
        "top_format_label": top_format_label,

        # expected by the **new template**
        "best_type": best_type,   # 'courses' | 'books' | 'videos'
        "top": top,               # list of dicts (≤2)
        "supporting": supporting, # dict of lists (≤2 each)
        "surprise": surprise,     # dict or None
    })


@require_http_methods(["GET", "POST"])
def landing(request):
    initial = {}
    if request.user.is_authenticated:
        try:
            profile = LearnerProfile.objects.get(user=request.user)
            # assume profile.topic_interests is a list/array of strings
            initial["topic_interests"] = profile.topic_interests or []
        except LearnerProfile.DoesNotExist:
            profile = None
    else:
        profile = None

    if request.method == "POST":
        if not request.user.is_authenticated:
            # not logged in: nudge to register/login while preserving choices in session
            request.session["pending_topic_interests"] = request.POST.getlist("topic_interests")
            return redirect("register")
        form = TopicSelectionForm(request.POST)
        if form.is_valid():
            topics = form.cleaned_data["topic_interests"]
            profile, _ = LearnerProfile.objects.get_or_create(user=request.user)
            profile.topic_interests = topics
            profile.save()
            messages.success(request, "Your topics were saved. Here are some picks!")
            return redirect("dashboard")
    else:
        # prefill from profile or from any session-cached choice (e.g., from pre-auth selection)
        pending = request.session.pop("pending_topic_interests", None)
        if pending:
            initial["topic_interests"] = pending
        form = TopicSelectionForm(initial=initial)

    return render(request, "core/landing.html", {
        "form": form,
        "profile": profile,
    })
