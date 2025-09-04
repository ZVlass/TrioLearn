from apps.core.models import Course
import csv
from django.core.management.base import BaseCommand
from tqdm import tqdm

class Command(BaseCommand):
    help = "Import courses data from CSV (and delete existing first)"

    def handle(self, *args, **kwargs):

        # file path 
        file_path = "../data/interim/courses_metadata.csv"

        # Delete existing data
        Course.objects.all().delete()
        self.stdout.write("Existing courses deleted.")

        # Count rows first
        with open(file_path, newline='', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1  

        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                Course.objects.get_or_create(
                    title=row["Title"],
                    defaults={
                        "ratings": row.get("Ratings") or None,
                        "rating_level": row.get("rating_level"),
                        "difficulty_num": row.get("difficulty_num") or None,
                        "students_enrolled": row.get("students_enrolled") or 0,
                        "popularity": row.get("popularity") or 0.0,
                        "platform": "Coursera"
                    }
                )

        self.stdout.write(self.style.SUCCESS("Courses import complete"))


