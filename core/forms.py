from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator
from .models import *
from django.utils import timezone


class FarmerRegistrationForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(required=True)
    phone = forms.CharField(max_length=15, required=True)
    address = forms.CharField(widget=forms.Textarea(attrs={'rows': 3}), required=True)
    
    # Farm details
    farm_name = forms.CharField(max_length=200, required=True)
    farm_size_acres = forms.FloatField(validators=[MinValueValidator(0.1)], required=True)
    crops_grown = forms.CharField(widget=forms.Textarea(attrs={'rows': 2}), 
                                help_text="List crops you grow (comma separated)")
    
    # Location (will be populated by JavaScript)
    latitude = forms.FloatField(widget=forms.HiddenInput(), required=False)
    longitude = forms.FloatField(widget=forms.HiddenInput(), required=False)
    
    # Language preference
    LANGUAGE_CHOICES = [
        ('en', 'English'),
        ('hi', 'Hindi'),
        ('te', 'Telugu'),
        ('ta', 'Tamil'),
    ]
    preferred_language = forms.ChoiceField(choices=LANGUAGE_CHOICES, initial='en')
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields:
            self.fields[field].widget.attrs.update({'class': 'form-control'})
        
        # Add Bootstrap classes and placeholders
        self.fields['username'].widget.attrs.update({'placeholder': 'Enter username'})
        self.fields['email'].widget.attrs.update({'placeholder': 'Enter email address'})
        self.fields['phone'].widget.attrs.update({'placeholder': 'Enter phone number'})
        self.fields['farm_name'].widget.attrs.update({'placeholder': 'Enter your farm name'})
        self.fields['farm_size_acres'].widget.attrs.update({'placeholder': 'Farm size in acres'})

class WorkerRegistrationForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(required=True)
    phone = forms.CharField(max_length=15, required=True)
    address = forms.CharField(widget=forms.Textarea(attrs={'rows': 3}), required=True)
    
    # Worker details
    service_radius_km = forms.IntegerField(
        min_value=5, max_value=100, initial=10,
        help_text="Service radius in kilometers"
    )
    hourly_rate = forms.DecimalField(
        max_digits=8, decimal_places=2, validators=[MinValueValidator(50)],
        help_text="Your hourly rate in INR"
    )
    experience_years = forms.IntegerField(
        min_value=0, max_value=50, initial=0,
        help_text="Years of experience in agricultural services"
    )
    equipment_owned = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 2}),
        help_text="List equipment you own (comma separated)"
    )
    
    # Location
    latitude = forms.FloatField(widget=forms.HiddenInput(), required=False)
    longitude = forms.FloatField(widget=forms.HiddenInput(), required=False)
    
    # Language preference
    LANGUAGE_CHOICES = [
        ('en', 'English'),
        ('hi', 'Hindi'),
        ('te', 'Telugu'),
        ('ta', 'Tamil'),
    ]
    preferred_language = forms.ChoiceField(choices=LANGUAGE_CHOICES, initial='en')
    
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields:
            self.fields[field].widget.attrs.update({'class': 'form-control'})

class DiseaseDetectionForm(forms.ModelForm):
    CROP_CHOICES = [
        ('tomato', 'Tomato'),
        ('potato', 'Potato'),
        ('corn', 'Corn'),
        ('wheat', 'Wheat'),
        ('rice', 'Rice'),
        ('cotton', 'Cotton'),
        ('soybean', 'Soybean'),
        ('apple', 'Apple'),
        ('grape', 'Grape'),
        ('other', 'Other'),
    ]
    
    crop_type = forms.ChoiceField(
        choices=CROP_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    crop_image = forms.ImageField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*',
            'capture': 'camera'  # Enables camera capture on mobile
        }),
        help_text="Upload a clear image of the affected crop"
    )
    
    class Meta:
        model = CropDiseaseDetection
        fields = ['crop_type', 'crop_image']

class ServiceBookingForm(forms.ModelForm):
    CROP_CHOICES = [
        ('tomato', 'Tomato'),
        ('potato', 'Potato'),
        ('corn', 'Corn'),
        ('wheat', 'Wheat'),
        ('rice', 'Rice'),
        ('cotton', 'Cotton'),
        ('soybean', 'Soybean'),
        ('apple', 'Apple'),
        ('grape', 'Grape'),
        ('other', 'Other'),
    ]
    
    crop_type = forms.ChoiceField(
        choices=CROP_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    area_acres = forms.FloatField(
        validators=[MinValueValidator(0.1)],
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0.1',
            'placeholder': 'Area to be treated (acres)'
        })
    )
    
    preferred_date = forms.DateTimeField(
        widget=forms.DateTimeInput(attrs={
            'class': 'form-control',
            'type': 'datetime-local',
            'min': timezone.now().strftime('%Y-%m-%dT%H:%M')
        }),
        help_text="Select preferred date and time for service"
    )
    
    farm_latitude = forms.FloatField(widget=forms.HiddenInput())
    farm_longitude = forms.FloatField(widget=forms.HiddenInput())
    
    farm_address = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Enter detailed farm address'
        })
    )
    
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Any additional notes or special instructions'
        })
    )
    
    class Meta:
        model = ServiceBooking
        fields = ['crop_type', 'area_acres', 'preferred_date', 'farm_latitude', 
                 'farm_longitude', 'farm_address', 'notes']

class ProfileUpdateForm(forms.ModelForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(required=True)
    
    LANGUAGE_CHOICES = [
        ('en', 'English'),
        ('hi', 'Hindi'),
        ('te', 'Telugu'),
        ('ta', 'Tamil'),
    ]
    preferred_language = forms.ChoiceField(choices=LANGUAGE_CHOICES)
    
    class Meta:
        model = UserProfile
        fields = ['phone', 'address', 'preferred_language', 'profile_image']
        widgets = {
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
            'address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'preferred_language': forms.Select(attrs={'class': 'form-control'}),
            'profile_image': forms.FileInput(attrs={'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.user:
            self.fields['first_name'].initial = self.instance.user.first_name
            self.fields['last_name'].initial = self.instance.user.last_name
            self.fields['email'].initial = self.instance.user.email
        
        for field in ['first_name', 'last_name', 'email']:
            self.fields[field].widget.attrs.update({'class': 'form-control'})
    
    def save(self, commit=True):
        profile = super().save(commit=False)
        if commit:
            # Update user fields
            profile.user.first_name = self.cleaned_data['first_name']
            profile.user.last_name = self.cleaned_data['last_name']
            profile.user.email = self.cleaned_data['email']
            profile.user.save()
            profile.save()
        return profile

class WorkerAvailabilityForm(forms.ModelForm):
    class Meta:
        model = WorkerProfile
        fields = ['is_available', 'service_radius_km', 'hourly_rate']
        widgets = {
            'is_available': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'service_radius_km': forms.NumberInput(attrs={'class': 'form-control', 'min': '5', 'max': '100'}),
            'hourly_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'min': '50'}),
        }

class BookingFilterForm(forms.Form):
    STATUS_CHOICES = [
        ('', 'All Status'),
        ('pending', 'Pending'),
        ('accepted', 'Accepted'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]
    
    PAYMENT_STATUS_CHOICES = [
        ('', 'All Payments'),
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('refunded', 'Refunded'),
    ]
    
    status = forms.ChoiceField(
        choices=STATUS_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    payment_status = forms.ChoiceField(
        choices=PAYMENT_STATUS_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    date_from = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    
    date_to = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )

class ReviewForm(forms.ModelForm):
    class Meta:
        model = WorkerReview
        fields = ['rating', 'comment']
        widgets = {
            'rating': forms.RadioSelect(choices=[(i, i) for i in range(1, 6)]),
            'comment': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Share your experience with this worker...'
            }),
        }