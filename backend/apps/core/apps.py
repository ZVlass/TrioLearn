from django.apps import AppConfig


class CoursesConfig(AppConfig):
    name = "apps.core"
    label = "core"
    verbose_name = "core"

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.core'
    
