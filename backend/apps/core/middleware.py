import time
from datetime import timedelta
from django.utils.deprecation import MiddlewareMixin
from .models import LearnerProfile

class SessionDurationMiddleware(MiddlewareMixin):
    def process_request(self, request):
        if request.user.is_authenticated:
            now = time.time()
            last_time = request.session.get('last_active_time', now)
            session_duration = now - last_time

            # Update session time
            request.session['last_active_time'] = now

            # Update profile average (optional: use exponential moving average)
            if session_duration > 2:  # Ignore refreshes under 2s
                profile = LearnerProfile.objects.filter(user=request.user).first()
                if profile:
                    if not hasattr(request, "_session_total"):
                        profile.avg_session_duration_min = (
                            (profile.avg_session_duration_min or 0) + (session_duration / 60)
                        ) / 2  # Simple averaging
                        profile.save()
