
# Create your models here.
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from datetime import datetime
import uuid
import json

class UserProfile(models.Model):
    USER_TYPES = [
        ('farmer', 'Farmer'),
        ('worker', 'Spray Worker'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    user_type = models.CharField(max_length=10, choices=USER_TYPES)
    phone = models.CharField(max_length=15)
    address = models.TextField()
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    profile_image = models.ImageField(upload_to='profiles/', null=True, blank=True)
    preferred_language = models.CharField(max_length=5, default='en')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # AI Assistant preferences
    voice_enabled = models.BooleanField(default=True)
    auto_translate = models.BooleanField(default=True)
    auto_speak = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.user.username} - {self.user_type}"

class FarmerProfile(models.Model):
    user_profile = models.OneToOneField(UserProfile, on_delete=models.CASCADE)
    farm_name = models.CharField(max_length=200)
    farm_size_acres = models.FloatField(validators=[MinValueValidator(0.1)])
    crops_grown = models.TextField(help_text="List of crops grown")
    farming_experience_years = models.IntegerField(default=0)
    preferred_farming_method = models.CharField(max_length=50, choices=[
        ('organic', 'Organic'),
        ('conventional', 'Conventional'),
        ('mixed', 'Mixed')
    ], default='conventional')
    
    def __str__(self):
        return f"{self.farm_name} - {self.user_profile.user.username}"

class WorkerProfile(models.Model):
    user_profile = models.OneToOneField(UserProfile, on_delete=models.CASCADE)
    service_radius_km = models.IntegerField(default=10)
    hourly_rate = models.DecimalField(max_digits=8, decimal_places=2)
    experience_years = models.IntegerField(default=0)
    equipment_owned = models.TextField(help_text="List of equipment owned")
    is_available = models.BooleanField(default=True)
    rating = models.FloatField(default=0.0, validators=[MinValueValidator(0), MaxValueValidator(5)])
    total_jobs = models.IntegerField(default=0)
    total_earnings = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    
    # Service specializations
    specializations = models.JSONField(default=list, blank=True)  # ['pesticide_spray', 'fertilizer_application']
    certifications = models.TextField(blank=True)
    
    def __str__(self):
        return f"Worker - {self.user_profile.user.username}"

class CropDiseaseDetection(models.Model):
    DISEASE_STATUS = [
        ('healthy', 'Healthy'),
        ('diseased', 'Diseased'),
        ('unknown', 'Unknown'),
    ]
    
    SEVERITY_LEVELS = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]
    
    farmer = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    crop_image = models.ImageField(upload_to='crop_images/')
    crop_type = models.CharField(max_length=100)
    
    # Basic detection results
    disease_detected = models.CharField(max_length=200, null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=DISEASE_STATUS, default='unknown')
    suggestions = models.TextField(null=True, blank=True)
    
    # Enhanced AI analysis results
    ai_analysis = models.JSONField(default=dict, blank=True)  # Store full Gemini analysis
    plant_id = models.CharField(max_length=100, blank=True)  # Scientific plant ID
    disease_id = models.CharField(max_length=100, blank=True)  # Scientific disease ID
    severity = models.CharField(max_length=10, choices=SEVERITY_LEVELS, default='medium')
    urgency = models.CharField(max_length=20, default='routine')  # immediate/within_week/routine
    
    # Translated content for multilingual support
    suggestions_translated = models.JSONField(null=True, blank=True)
    
    # Analysis metadata
    analysis_method = models.CharField(max_length=50, default='gemini_ai')
    processing_time = models.FloatField(null=True, blank=True)  # seconds
    detected_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-detected_at']
    
    def __str__(self):
        return f"{self.crop_type} - {self.disease_detected or 'No disease'}"
    
    def get_severity_color(self):
        """Return Bootstrap color class based on severity"""
        colors = {
            'low': 'success',
            'medium': 'warning', 
            'high': 'danger'
        }
        return colors.get(self.severity, 'info')
    
    def get_confidence_percentage(self):
        """Return confidence as percentage string"""
        if self.confidence_score:
            return f"{self.confidence_score:.1f}%"
        return "N/A"

class ServiceBooking(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('accepted', 'Accepted'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]
    
    PAYMENT_STATUS = [
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('refunded', 'Refunded'),
    ]
    
    URGENCY_LEVELS = [
        ('routine', 'Routine'),
        ('within_week', 'Within Week'),
        ('immediate', 'Immediate'),
    ]
    
    booking_id = models.UUIDField(default=uuid.uuid4, unique=True)
    farmer = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='farmer_bookings')
    worker = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='worker_bookings', null=True, blank=True)
    
    # Service Details
    service_type = models.CharField(max_length=100, default='Pesticide Spray')
    crop_type = models.CharField(max_length=100)
    area_acres = models.FloatField(validators=[MinValueValidator(0.1)])
    disease_detection = models.ForeignKey(CropDiseaseDetection, on_delete=models.SET_NULL, null=True, blank=True)
    urgency = models.CharField(max_length=20, choices=URGENCY_LEVELS, default='routine')
    
    # Location
    farm_latitude = models.FloatField()
    farm_longitude = models.FloatField()
    farm_address = models.TextField()
    
    # Booking Details
    preferred_date = models.DateTimeField()
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='pending')
    notes = models.TextField(blank=True)
    
    # Payment
    estimated_cost = models.DecimalField(max_digits=10, decimal_places=2)
    final_cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    payment_status = models.CharField(max_length=10, choices=PAYMENT_STATUS, default='pending')
    razorpay_order_id = models.CharField(max_length=100, null=True, blank=True)
    razorpay_payment_id = models.CharField(max_length=100, null=True, blank=True)
    
    # Service completion details
    work_started_at = models.DateTimeField(null=True, blank=True)
    work_completed_at = models.DateTimeField(null=True, blank=True)
    actual_duration_hours = models.FloatField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    accepted_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Booking {self.booking_id} - {self.farmer.user.username}"
    
    def get_status_color(self):
        """Return Bootstrap color class based on status"""
        colors = {
            'pending': 'warning',
            'accepted': 'info',
            'in_progress': 'primary',
            'completed': 'success',
            'cancelled': 'danger'
        }
        return colors.get(self.status, 'secondary')
    
    def get_urgency_color(self):
        """Return Bootstrap color class based on urgency"""
        colors = {
            'routine': 'success',
            'within_week': 'warning',
            'immediate': 'danger'
        }
        return colors.get(self.urgency, 'info')

class BookingStatusUpdate(models.Model):
    booking = models.ForeignKey(ServiceBooking, on_delete=models.CASCADE, related_name='status_updates')
    status = models.CharField(max_length=15)
    notes = models.TextField(blank=True)
    updated_by = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    location_lat = models.FloatField(null=True, blank=True)  # Worker's location when updating
    location_lng = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']

class WorkerReview(models.Model):
    booking = models.OneToOneField(ServiceBooking, on_delete=models.CASCADE)
    farmer = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='reviews_given')
    worker = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='reviews_received')
    rating = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)])
    comment = models.TextField(blank=True)
    
    # Detailed ratings
    punctuality_rating = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)], null=True)
    quality_rating = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)], null=True)
    communication_rating = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)], null=True)
    
    # Review metadata
    is_verified = models.BooleanField(default=True)  # Based on completed booking
    helpful_votes = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Review for {self.worker.user.username} - {self.rating}/5"
    
    def get_average_detailed_rating(self):
        """Calculate average of detailed ratings"""
        ratings = [self.punctuality_rating, self.quality_rating, self.communication_rating]
        valid_ratings = [r for r in ratings if r is not None]
        return sum(valid_ratings) / len(valid_ratings) if valid_ratings else self.rating

class Notification(models.Model):
    NOTIFICATION_TYPES = [
        ('booking_created', 'Booking Created'),
        ('booking_accepted', 'Booking Accepted'),
        ('booking_completed', 'Booking Completed'),
        ('booking_cancelled', 'Booking Cancelled'),
        ('payment_success', 'Payment Success'),
        ('disease_detected', 'Disease Detected'),
        ('worker_nearby', 'Worker Nearby'),
        ('system_update', 'System Update'),
        ('weather_alert', 'Weather Alert'),
    ]
    
    PRIORITY_LEVELS = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    ]
    
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    message = models.TextField()
    notification_type = models.CharField(max_length=20, choices=NOTIFICATION_TYPES)
    priority = models.CharField(max_length=10, choices=PRIORITY_LEVELS, default='medium')
    
    # Related objects
    related_booking = models.ForeignKey(ServiceBooking, on_delete=models.CASCADE, null=True, blank=True)
    related_detection = models.ForeignKey(CropDiseaseDetection, on_delete=models.CASCADE, null=True, blank=True)
    
    # Notification status
    is_read = models.BooleanField(default=False)
    is_sent = models.BooleanField(default=False)  # For email/SMS notifications
    read_at = models.DateTimeField(null=True, blank=True)
    
    # Delivery channels
    send_email = models.BooleanField(default=True)
    send_sms = models.BooleanField(default=False)
    send_push = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} - {self.user.user.username}"
    
    def mark_as_read(self):
        """Mark notification as read"""
        if not self.is_read:
            self.is_read = True
            self.read_at = datetime.now()
            self.save()
    

class ChatHistory(models.Model):
    """Store chat history with AI assistant"""
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    context = models.TextField(blank=True)  # Context provided with the message
    language = models.CharField(max_length=5, default='en')
    
    # AI metadata
    model_used = models.CharField(max_length=50, default='gemini-1.5-pro')
    confidence_score = models.FloatField(null=True, blank=True)
    response_time = models.FloatField(null=True, blank=True)  # seconds
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Chat histories"

class CropRecommendation(models.Model):
    """Store crop recommendations for different seasons and locations"""
    location_lat = models.FloatField()
    location_lng = models.FloatField()
    season = models.CharField(max_length=20)  # rabi, kharif, summer
    
    # Recommendation data
    recommended_crops = models.JSONField()  # List of crop recommendations
    climate_data = models.JSONField(null=True, blank=True)
    soil_recommendations = models.JSONField(null=True, blank=True)
    
    # Metadata
    generated_by = models.CharField(max_length=50, default='gemini_ai')
    confidence_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

class WeatherAlert(models.Model):
    """Weather alerts affecting farming activities"""
    ALERT_TYPES = [
        ('rain', 'Heavy Rain'),
        ('drought', 'Drought Warning'),
        ('hail', 'Hail Storm'),
        ('frost', 'Frost Warning'),
        ('heat', 'Heat Wave'),
        ('wind', 'Strong Winds'),
    ]
    
    SEVERITY_LEVELS = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('extreme', 'Extreme'),
    ]
    
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    severity = models.CharField(max_length=10, choices=SEVERITY_LEVELS)
    title = models.CharField(max_length=200)
    description = models.TextField()
    
    # Affected area
    region = models.CharField(max_length=100)
    coordinates = models.JSONField()  # Polygon coordinates
    
    # Timing
    starts_at = models.DateTimeField()
    ends_at = models.DateTimeField()
    
    # Recommendations
    farming_advice = models.TextField()
    precautions = models.JSONField(default=list)
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']


# Custom model managers for common queries
class ActiveBookingsManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().exclude(status__in=['completed', 'cancelled'])

class CompletedBookingsManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(status='completed')

class UnreadNotificationsManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(is_read=False)

# Add managers to models
ServiceBooking.add_to_class('active_bookings', ActiveBookingsManager())
ServiceBooking.add_to_class('completed_bookings', CompletedBookingsManager())
Notification.add_to_class('unread', UnreadNotificationsManager())















