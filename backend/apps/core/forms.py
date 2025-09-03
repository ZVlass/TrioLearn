from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .models import LearnerProfile

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
IMD_BAND_CHOICES = [
    ('0-10%', '0-10%'), ('10-20%', '10-20%'), ('20-30%', '20-30%'),
    ('30-40%', '30-40%'), ('40-50%', '40-50%'), ('50-60%', '50-60%'),
    ('60-70%', '60-70%'), ('70-80%', '70-80%'), ('80-90%', '80-90%'),
    ('90-100%', '90-100%')
]
AGE_BAND_CHOICES = [
    ('0-35', '0-35'),
    ('35-55', '35-55'),
    ('55<=', '55+'),
]

TOPIC_CHOICES = [
    ('ai', 'AI / Machine Learning'),
    ('programming', 'Programming'),
    ('finance', 'Business / Finance'),
    ('medicine', 'Health & Medicine'),
    ('arts', 'Arts & Humanities'),
    ('datasci', 'Data Science'),
    ('languages', 'Language Learning'),
]

FORMAT_CHOICES = [
    ('video', 'Watching videos'),
    ('course', 'Interactive courses'),
    ('reading', 'Reading books/articles'),
    ('none', 'No preference'),
]

class UserRegistrationForm(forms.Form):
    username = forms.CharField(max_length=150)
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)

    gender = forms.ChoiceField(choices=GENDER_CHOICES)
    region = forms.ChoiceField(choices=REGION_CHOICES)
    highest_education = forms.ChoiceField(choices=EDUCATION_CHOICES)
    age_band = forms.ChoiceField(choices=AGE_BAND_CHOICES)
    

    topic_interests = forms.MultipleChoiceField(
        choices=TOPIC_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        required=True
    )
    preferred_format = forms.ChoiceField(
        choices=FORMAT_CHOICES,
        widget=forms.RadioSelect,
        required=True
    )

    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise ValidationError("Email already in use.")
        return email
    
    def clean_topic_interests(self):
        topics = self.cleaned_data['topic_interests']
        if len(topics) > 3:
            raise ValidationError("Please select up to 3 topics only.")
        return topics