
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

            profile = LearnerProfile.objects.create(
                user=user,
                gender=form.cleaned_data['gender'],
                region=form.cleaned_data['region'],
                highest_education=form.cleaned_data['highest_education'],
                imd_band=form.cleaned_data['imd_band'],
                age_band=form.cleaned_data['age_band'],
                avg_session_duration_min=form.cleaned_data.get('avg_session_duration_min', None),
                course_prop=0.0,
                reading_prop=0.0,
                video_prop=0.0,
            )

            login(request, user)
            messages.success(request, "Welcome to TrioLearn!")
            return redirect('dashboard')  # Replace with actual view name
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
    return render(request, 'core/dashboard.html', {'profile': profile})