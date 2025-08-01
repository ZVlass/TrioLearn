import csv
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from apps.core.models import LearnerProfile
from tqdm import tqdm

class Command(BaseCommand):
    help = "Import learner profiles and users from OULAD CSV"

    def handle(self, *args, **kwargs):
        path = "../data/processed/oulad_media_profiles_refined_balanced.csv"

        # Wipe old data
        LearnerProfile.objects.all().delete()
        User.objects.filter(username__startswith="user_").delete()
        self.stdout.write("Existing LearnerProfiles and test Users deleted.")

        # Count total rows
        with open(path, newline='', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1

        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, total=total_rows, desc="👤 Importing learners"):
                user, _ = User.objects.get_or_create(
                    username=f"user_{row['id_student']}",
                    defaults={
                        "email": f"user_{row['id_student']}@example.com",
                        "password": "pbkdf2_sha256$260000$placeholder",  # No usable password
                    }
                )

                LearnerProfile.objects.get_or_create(
                    user=user,
                    defaults={
                        "gender": row.get("gender"),
                        "region": row.get("region"),
                        "highest_education": row.get("highest_education"),
                        "imd_band": row.get("imd_band"),
                        "age_band": row.get("age_band"),
                        "avg_session_duration_min": float(row.get("avg_session_duration_min") or 0),
                        "course_prop": float(row.get("course_prop") or 0),
                        "reading_prop": float(row.get("reading_prop") or 0),
                        "video_prop": float(row.get("video_prop") or 0),
                    }
                )

        self.stdout.write(self.style.SUCCESS("Learner import complete"))
