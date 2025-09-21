from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from rest_framework import viewsets

from .forms import UserRegistrationForm, TopicSelectionForm
from .models import LearnerProfile, Course, Book, Video, Interaction
from .serializers import (
    LearnerProfileSerializer,
    CourseSerializer,
    BookSerializer,
    VideoSerializer,
    InteractionSerializer
)

# Prefer these if you have them (legacy/new engine)
try:
    from apps.recommender.engine import get_recommendations, recommend_query  # legacy/desired
except Exception:  # keep the import from blowing up in dev
    def get_recommendations(*args, **kwargs):
        raise TypeError("get_recommendations placeholder")
    def recommend_query(*args, **kwargs):
        raise TypeError("recommend_query placeholder")

# Always keep a safe fallback available (present in your codebase now)
from apps.recommender.utils import recommend_for_profile  # fallback path. :contentReference[oaicite:2]{index=2}

from apps.core.constants import TOPIC_LABELS


# -------------------------
# Auth endpoints (unchanged)
# -------------------------
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
            user = User.objects.create_user(
                username=data['username'],
                email=data['email'],
                password=data['password'],
            )
            user.first_name = (data.get('first_name') or '').strip()
            user.last_name  = (data.get('last_name') or '').strip()
            user.save()

            preferred = (data.get('preferred_format') or 'none').strip().lower()

            # Simple soft-priors
            course_prop = reading_prop = video_prop = 0.33
            if preferred in ('course','reading','video'):
                if preferred == 'course': course_prop = 0.5
                elif preferred == 'reading': reading_prop = 0.5
                else: video_prop = 0.5

            LearnerProfile.objects.create(
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
            )

            login(request, user)
            messages.success(request, "Welcome to TrioLearn!")
            return redirect('dashboard')
    else:
        form = UserRegistrationForm()
    return render(request, 'core/register.html', {'form': form})


def login_user(request):
    if request.method == 'POST':
        username = request.POST.get('username') or ''
        password = request.POST.get('password') or ''
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('dashboard')
        messages.error(request, 'Invalid username or password.')
    return render(request, 'core/login.html')


def logout_user(request):
    logout(request)
    messages.success(request, "You have successfully logged out.")
    return redirect('')


# -------------------------
# Landing: pick topics (unchanged)
# -------------------------
@require_http_methods(["GET", "POST"])
def landing(request):
    initial = {}
    profile = None
    if request.user.is_authenticated:
        try:
            profile = LearnerProfile.objects.get(user=request.user)
            initial["topic_interests"] = profile.topic_interests or []
        except LearnerProfile.DoesNotExist:
            profile = None

    if request.method == "POST":
        if not request.user.is_authenticated:
            request.session["pending_topic_interests"] = request.POST.getlist("topic_interests")
            return redirect("register")
        form = TopicSelectionForm(request.POST)
        if form.is_valid():
            topics = form.cleaned_data["topic_interests"]
            profile, _ = LearnerProfile.objects.get_or_create(user=request.user)
            profile.topic_interests = topics
            profile.save()
            messages.success(request, "Your topics were saved.")
            return redirect("dashboard")
    else:
        pending = request.session.pop("pending_topic_interests", None)
        if pending:
            initial["topic_interests"] = pending
        form = TopicSelectionForm(initial=initial)

    return render(request, "core/landing.html", {"form": form, "profile": profile})


# -------------------------
# Dashboard — uses recommend_query AND get_recommendations
# -------------------------
@login_required
def dashboard(request):
    profile = LearnerProfile.objects.get(user=request.user)

    # Optional free-text query and focus topic
    q = (request.GET.get("q") or "").strip() or None
    focus_topic = (request.GET.get("topic") or "").strip() or None
    if focus_topic and profile.topic_interests and focus_topic not in (profile.topic_interests or []):
        focus_topic = None

    # Topic chips + label for UI
    topic_display = [(c, TOPIC_LABELS.get(c, c)) for c in (profile.topic_interests or [])]
    focus_label = TOPIC_LABELS.get(focus_topic) if focus_topic else None

    # Engine fetch size (keep larger so we have leftovers)
    try:
        top_k = int(request.GET.get("top_k", 12))
    except (TypeError, ValueError):
        top_k = 12

    # Items to DISPLAY per box
    try:
        per_box = int(request.GET.get("per_box", 5))
    except (TypeError, ValueError):
        per_box = 5

    # Pick the query for the engine: priority q → focus label → default
    if q:
        query = q
    elif focus_topic:
        query = focus_label or focus_topic
    else:
        query = "machine learning"

    # Level heuristic
    level = (profile.highest_education or "").strip().lower() or "intermediate"

    # === Vector engine ===
    # Returns dict with keys: "books", "courses", "videos", "best_type", "surprise"
    recs = get_recommendations(query=query, level=level, top_k=top_k)

    # Full lists from engine
    courses_all = list(recs.get("courses", []))
    books_all   = list(recs.get("books", []))
    videos_all  = list(recs.get("videos", []))

    # Determine primary modality (fallback to largest list)
    best_type = recs.get("best_type")
    if not best_type:
        sizes = {"courses": len(courses_all), "books": len(books_all), "videos": len(videos_all)}
        best_type = max(sizes, key=sizes.get) if any(sizes.values()) else "courses"

    # Slice what we SHOW (per_box)
    courses = courses_all[:per_box]
    books   = books_all[:per_box]
    videos  = videos_all[:per_box]

    top = {"courses": courses, "books": books, "videos": videos}.get(best_type, [])
    supporting = {
        "courses": courses if best_type != "courses" else [],
        "books": books if best_type != "books" else [],
        "videos": videos if best_type != "videos" else [],
    }

    # Build leftovers (not displayed) to choose a genuine surprise from
    leftovers = {
        "courses": courses_all[per_box:],
        "books":   books_all[per_box:],
        "videos":  videos_all[per_box:],
    }

    # Surprise: prefer a non-primary modality with the largest leftover pool
    non_primary = [m for m in ("courses", "books", "videos") if m != best_type]
    non_primary.sort(key=lambda m: len(leftovers[m]), reverse=True)

    surprise = None
    for m in non_primary:
        if leftovers[m]:
            surprise = leftovers[m][0]     # deterministic; or use random.choice(leftovers[m])
            break
    if surprise is None and leftovers[best_type]:
        surprise = leftovers[best_type][0]

    # Normalize surprise for template compatibility
    def _norm_surprise(s):
        # Engine returns dicts; keep fallback if model instance ever appears
        if not isinstance(s, dict):
            return s
        s = dict(s)
        s.setdefault("url", s.get("href") or s.get("info_link") or "#")
        s.setdefault("title", s.get("display_title") or s.get("title") or "Untitled")
        s.setdefault("id", s.get("display_id") or s.get("id") or "")
        return s

    surprise = _norm_surprise(surprise)

    # Derive surprise_type (used by your template’s old/new surprise blocks)
    surprise_type = None
    if isinstance(surprise, dict):
        mod = (surprise.get("modality") or "").lower()
        if mod in ("course", "courses"): surprise_type = "course"
        elif mod in ("book", "books"):   surprise_type = "book"
        elif mod in ("video", "videos"): surprise_type = "video"
    elif surprise:
        if isinstance(surprise, Course): surprise_type = "course"
        elif isinstance(surprise, Book): surprise_type = "book"
        elif isinstance(surprise, Video): surprise_type = "video"

    # Debug log
    print(
        "[recs] (engine)",
        "query=", query,
        f"shown(c/b/v)={[len(courses), len(books), len(videos)]}",
        f"leftovers(c/b/v)={[len(leftovers['courses']), len(leftovers['books']), len(leftovers['videos'])]}",
        "surprise_mod=", (surprise.get('modality') if isinstance(surprise, dict) else type(surprise).__name__),
    )

    # Banner labels
    FORMAT_LABELS = {
        "video": "Videos",
        "course": "Courses",
        "reading": "Books & Articles",
        "none": "No preference",
    }
    preferred_format_label = FORMAT_LABELS.get((profile.preferred_format or "none"))

    if hasattr(profile, "top_format_key_and_label"):
        _, top_format_label = profile.top_format_key_and_label()
    else:
        top_format_label = {"courses": "Courses", "books": "Books & Articles", "videos": "Videos"}.get(best_type, "Courses")

    GENDER_LABELS = {"M": "Male", "F": "Female", "O": "Other"}
    gender_label = GENDER_LABELS.get(profile.gender, profile.gender or "")

    return render(request, "core/dashboard.html", {
        "profile": profile,
        "gender_label": gender_label,
        "topic_display": topic_display,
        "focus_topic": focus_topic,
        "focus_label": focus_label,
        "preferred_format_label": preferred_format_label,
        "top_format_label": top_format_label,
        "best_type": best_type,
        "top": top,
        "supporting": supporting,
        "surprise": surprise,
        "surprise_type": surprise_type,
        "q": q,
    })