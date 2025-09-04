import csv
from django.core.management.base import BaseCommand
from apps.core.models import Book
from tqdm import tqdm

class Command(BaseCommand):
    help = "Import book data from CSV (and delete existing first)"

    def handle(self, *args, **kwargs):

        # file path 
        file_path = "../data/interim/books_metadata.csv"

        # Delete existing data
        Book.objects.all().delete()
        self.stdout.write("Existing books deleted.")

        # Count rows first
        with open(file_path, newline='', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1  

        with open(file_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, total=total_rows, desc="📚 Importing books"):
                Book.objects.create(
                    keyword=row.get("keyword"),
                    title=row["title"],
                    authors=row.get("authors"),
                    description=row.get("description"),
                    categories=row.get("categories", ""),
                    published_date=row.get("publishedDate", ""),
                    page_count=row.get("pageCount") or 0,
                    language=row.get("language", ""),
                    average_rating=row.get("averageRating") or None,
                    ratings_count=int(float(row.get("ratingsCount") or 0)),
                    preview_link=row.get("previewLink", ""),
                    info_link=row.get("infoLink", "")
                )

        self.stdout.write(self.style.SUCCESS("Book import complete"))
