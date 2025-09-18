from django.contrib import admin
from .models import *

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'user_type', 'phone', 'preferred_language', 'created_at']
    list_filter = ['user_type', 'preferred_language', 'created_at']
    search_fields = ['user__username', 'user__email', 'phone']

@admin.register(FarmerProfile)
class FarmerProfileAdmin(admin.ModelAdmin):
    list_display = ['farm_name', 'user_profile', 'farm_size_acres']
    search_fields = ['farm_name', 'user_profile__user__username']

@admin.register(WorkerProfile)
class WorkerProfileAdmin(admin.ModelAdmin):
    list_display = ['user_profile', 'hourly_rate', 'experience_years', 'rating', 'is_available']
    list_filter = ['is_available', 'experience_years']
    search_fields = ['user_profile__user__username']

@admin.register(CropDiseaseDetection)
class CropDiseaseDetectionAdmin(admin.ModelAdmin):
    list_display = ['farmer', 'crop_type', 'disease_detected', 'status', 'confidence_score', 'detected_at']
    list_filter = ['status', 'crop_type', 'detected_at']
    search_fields = ['farmer__user__username', 'crop_type', 'disease_detected']
    readonly_fields = ['detected_at']

@admin.register(ServiceBooking)
class ServiceBookingAdmin(admin.ModelAdmin):
    list_display = ['booking_id', 'farmer', 'worker', 'status', 'payment_status', 'estimated_cost', 'created_at']
    list_filter = ['status', 'payment_status', 'service_type', 'created_at']
    search_fields = ['booking_id', 'farmer__user__username', 'worker__user__username']
    readonly_fields = ['booking_id', 'created_at', 'updated_at']

@admin.register(WorkerReview)
class WorkerReviewAdmin(admin.ModelAdmin):
    list_display = ['worker', 'farmer', 'rating', 'created_at']
    list_filter = ['rating', 'created_at']
    search_fields = ['worker__user__username', 'farmer__user__username']

@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ['user', 'title', 'notification_type', 'is_read', 'created_at']
    list_filter = ['notification_type', 'is_read', 'created_at']
    search_fields = ['user__user__username', 'title']

@admin.register(ChatHistory)
class ChatHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'message', 'language', 'created_at']
    list_filter = ['language', 'created_at']
    search_fields = ['user__user__username', 'message', 'response']

    
@admin.register(CropRecommendation)
class CropRecommendationAdmin(admin.ModelAdmin):
    list_display = ['season', 'recommended_crops', 'confidence_score', 'created_at']
    list_filter = ['season', 'created_at']
    search_fields = ['season', 'generated_by']



@admin.register(WeatherAlert)
class WeatherAlertAdmin(admin.ModelAdmin):
    list_display = ['region', 'alert_type', 'severity', 'created_at']
    list_filter = ['alert_type', 'severity', 'created_at']
    search_fields = ['region', 'alert_type', 'title']
    readonly_fields = ['created_at']
