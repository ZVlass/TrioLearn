
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm, TopicSelectionForm  
from apps.recommender.utils import recommend_for_profile
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
            user = User.objects.create_user(
                username=form.cleaned_data['username'],
                email=form.cleaned_data['email'],
                password=form.cleaned_data['password']
            )

            preferred = form.cleaned_data['preferred_format']
            # Initialize prop weights based on preferred format
            props = {'video': 0.6, 'reading': 0.6, 'course': 0.6}
            course_prop = props['course'] if preferred == 'course' else 0.2
            reading_prop = props['reading'] if preferred == 'reading' else 0.2
            video_prop = props['video'] if preferred == 'video' else 0.2

            profile = LearnerProfile.objects.create(
                user=user,
                gender=form.cleaned_data['gender'],
                region=form.cleaned_data['region'],
                highest_education=form.cleaned_data['highest_education'],
                age_band=form.cleaned_data['age_band'],
                topic_interests=form.cleaned_data['topic_interests'],
                preferred_format=preferred,
                course_prop=course_prop,
                reading_prop=reading_prop,
                video_prop=video_prop
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

@login_required
def dashboard(request):
    profile = LearnerProfile.objects.get(user=request.user)

    focus_topic = request.GET.get("topic") or None
    if focus_topic and profile.topic_interests and focus_topic not in profile.topic_interests:
        focus_topic = None

    topic_display = [(c, TOPIC_LABELS.get(c, c)) for c in (profile.topic_interests or [])]
    focus_label = TOPIC_LABELS.get(focus_topic) if focus_topic else None

    recs = recommend_for_profile(profile, k=12, explore_eps=0.3, focus_topic=focus_topic)
    top_key, top_label = profile.top_format_key_and_label()

    FORMAT_LABELS = {"video": "Videos", "course": "Courses", "reading": "Books & Articles", "none": "No preference"}
    preferred_format_label = FORMAT_LABELS.get((profile.preferred_format or "none"))

    # Full gender label for display (handles both codes and full strings)
    GENDER_LABELS = {"M": "Male", "F": "Female", "O": "Other", "Male": "Male", "Female": "Female", "Other": "Other"}
    gender_label = GENDER_LABELS.get(profile.gender, profile.gender or "")

    def to_item(obj):
        if isinstance(obj, Course):
            return {"type": "course", "id": obj.id, "title": obj.title, "url": obj.url}
        if isinstance(obj, Book):
            return {"type": "book", "id": obj.id, "title": obj.title, "url": obj.info_link}
        if isinstance(obj, Video):
            return {"type": "video", "id": obj.id, "title": obj.title, "url": obj.url}
        return None

    explore_items = [x for x in (to_item(y) for y in (recs.get("explore") or [])) if x]
    surprise_obj = recs.get("surprise")
    surprise_item = to_item(surprise_obj) if surprise_obj else None

    return render(request, "core/dashboard.html", {
        "profile": profile,
        "gender_label": gender_label,
        "topic_display": topic_display,
        "focus_topic": focus_topic,
        "focus_label": focus_label,
        "top_key": top_key,
        "top_format_label": top_label,
        "preferred_format_label": preferred_format_label,
        "courses": recs["courses"],
        "books": recs["books"],
        "videos": recs["videos"],
        "explore_items": explore_items,
        "surprise_item": surprise_item,
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
