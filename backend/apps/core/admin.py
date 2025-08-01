from django.contrib import admin

from .models import LearnerProfile, Course, Book, Video, Interaction

@admin.register(LearnerProfile)
class LearnerProfileAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'gender', 'region', 'highest_education',
        'age_band', 'course_prop', 'reading_prop', 'video_prop'
    )
    search_fields = ('user__username', 'region', 'highest_education')
    list_filter = ('gender', 'region', 'highest_education', 'age_band')

@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = (
        'title', 'platform', 'ratings', 'rating_level',
        'difficulty_num', 'students_enrolled', 'popularity'
    )
    search_fields = ('title', 'platform', 'rating_level')
    list_filter = ('platform', 'rating_level')

@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = (
        'title', 'authors', 'categories', 'language',
        'published_date', 'average_rating', 'ratings_count'
    )
    search_fields = ('title', 'authors', 'categories')
    list_filter = ('language', 'categories', 'published_date')

@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = (
        'title', 'channel', 'published_at', 'difficulty'
    )
    search_fields = ('title', 'channel')
    list_filter = ('channel', 'difficulty', 'published_at')

@admin.register(Interaction)
class InteractionAdmin(admin.ModelAdmin):
    list_display = (
        'learner', 'event_type', 'course', 'book', 'video', 'rating', 'timestamp'
    )
    search_fields = ('learner__user__username',)
    list_filter = ('event_type', 'timestamp')

    
