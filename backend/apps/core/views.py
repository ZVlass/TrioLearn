
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from .forms import UserRegistrationForm

from django.contrib.auth.decorators import login_required

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

    # Filter recommendations by topic match
    courses = [
        c for c in Course.objects.all()
        if profile.matches_item(c.topic_vector)
    ]

    books = [
        b for b in Book.objects.all()
        if profile.matches_item(b.topic_vector)
    ]

    videos = [
        v for v in Video.objects.all()
        if profile.matches_item(v.topic_vector)
    ]

    return render(request, 'core/dashboard.html', {
        'profile': profile,
        'courses': courses[:5],
        'books': books[:5],
        'videos': videos[:5],
    })