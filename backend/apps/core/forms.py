from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from apps.core.constants import TOPIC_CHOICES

GENDER_CHOICES = [('M', 'Male'), ('F', 'Female'), ('O', 'Other')]
REGION_CHOICES = [
    ('Outside the UK', 'Outside the UK' ),
    ('Prefer not to say', 'Prefer not to say'),
    ('East Anglian Region', 'East Anglian Region'),
    ('East Midlands Region', 'East Midlands Region'),
    ('Ireland', 'Ireland'),
    ('London Region', 'London Region'),
    ('North Region', 'North Region'),
    ('North Western Region', 'North Western Region'),
    ('Scotland', 'Scotland'),
    ('South East Region', 'South East Region'),
    ('South Region', 'South Region'),
    ('South West Region', 'South West Region'),
    ('Wales', 'Wales'),
    ('West Midlands Region', 'West Midlands Region'),
    ('Yorkshire Region', 'Yorkshire Region'),
]
EDUCATION_CHOICES = [
    ('No Formal quals', 'No Formal quals'),
    ('Lower Than A Level', 'Lower Than A Level'),
    ('A Level or Equivalent', 'A Level or Equivalent'),
    ('HE Qualification', 'HE Qualification'),
    ('Post Graduate Qualification', 'Post Graduate Qualification'),
]
AGE_BAND_CHOICES = [
    ('0-35', '0-35'),
    ('35-55', '35-55'),
    ('55<=', '55+'),
]

TOPIC_CHOICES = [
    ("ai", "Artificial Intelligence"),
    ("ml", "Machine Learning"),
    ("ds", "Data Science"),
    ("db", "Databases"),
    ("web", "Web Development"),
    ("nlp", "Natural Language Processing"),
    ("cv", "Computer Vision"),
    ("cloud", "Cloud/DevOps"),
]

FORMAT_CHOICES = [
    ('video', 'Watching videos'),
    ('course', 'Interactive courses'),
    ('reading', 'Reading books/articles'),
    ('none', 'No preference'),
]

class UserRegistrationForm(forms.Form):

    first_name = forms.CharField(max_length=150, required=False, label="First name")
    last_name  = forms.CharField(max_length=150, required=False, label="Last name")
    
    username = forms.CharField(max_length=150)
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)

    gender = forms.ChoiceField(choices=GENDER_CHOICES)
    region = forms.ChoiceField(choices=REGION_CHOICES)
    highest_education = forms.ChoiceField(choices=EDUCATION_CHOICES)
    age_band = forms.ChoiceField(choices=AGE_BAND_CHOICES)

    # Make topics optional; still enforce <= 3
    topic_interests = forms.MultipleChoiceField(
        choices=TOPIC_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        required=False,
        label="Topic interests (choose up to 3)"
    )

    preferred_format = forms.ChoiceField(
        choices=FORMAT_CHOICES,
        required=False,
        initial="none",
        widget=forms.RadioSelect,
        help_text="Prefer a format? You can skip — our recommendations adapt automatically."
    )

    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise ValidationError("Email already in use.")
        return email

    def clean_topic_interests(self):
        topics = self.cleaned_data.get('topic_interests', [])
        if len(topics) > 3:
            raise ValidationError("Please select up to 3 topics only.")
        return topics


class TopicSelectionForm(forms.Form):
    topic_interests = forms.MultipleChoiceField(
        choices=TOPIC_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        required=False,
        label="Choose up to 3 topics"
    )

    def clean_topic_interests(self):
        vals = self.cleaned_data.get("topic_interests", [])
        if len(vals) > 3:
            raise forms.ValidationError("Please select no more than 3 topics.")
        return vals
