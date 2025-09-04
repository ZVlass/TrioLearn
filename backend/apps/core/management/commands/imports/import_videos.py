from apps.core.models import Video
import csv
from django.core.management.base import BaseCommand
from tqdm import tqdm

class Command(BaseCommand):
    help = "Import videos data from CSV (and delete existing first)"

    def handle(self, *args, **kwargs):

        # file path 
        file_path = "../data/interim/videos_metadata.csv"

        # Delete existing data
        Video.objects.all().delete()
        self.stdout.write("Existing videos deleted.")

        # Count rows first
        with open(file_path, newline='', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1  

        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                Video.objects.get_or_create(
                    video_id=row["video_id"],
                    defaults={
                        "title": row.get("title"),
                        "description": row.get("description"),
                        "channel": row.get("channel"),
                        "published_at": row.get("published_at")
                    }
                )

        self.stdout.write(self.style.SUCCESS("Video import complete"))


