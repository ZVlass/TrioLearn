from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [
        ("core", "0004_book_gbooks_id_book_isbn_13_book_thumbnail"),
    ]
    operations = [migrations.RunSQL("CREATE EXTENSION IF NOT EXISTS vector;")]
