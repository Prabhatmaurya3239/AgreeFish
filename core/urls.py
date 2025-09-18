from django.urls import path, include
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Home and authentication
    path('', views.home, name='home'),
    path('register/farmer/', views.register_farmer, name='register_farmer'),
    path('register/worker/', views.register_worker, name='register_worker'),
    
    # Authentication URLs
    path('login/', auth_views.LoginView.as_view(
        template_name='registration/login.html',
        success_url='/dashboard/'
    ), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    
    # Dashboard URLs
    path('dashboard/', views.farmer_dashboard, name='farmer_dashboard'),
    path('worker/dashboard/', views.worker_dashboard, name='worker_dashboard'),
    
    # Disease Detection
    path('disease-detection/', views.disease_detection, name='disease_detection'),
    path('detection-result/<int:detection_id>/', views.detection_result, name='detection_result'),
    
    # Service Booking
    path('book-service/', views.book_service, name='book_service'),
    path('book-service/<int:detection_id>/', views.book_service, name='book_service_with_detection'),
    path('booking/payment/<int:booking_id>/', views.booking_payment, name='booking_payment'),
    path('payment/success/', views.payment_success, name='payment_success'),
    
    # Worker Actions
    path('accept-booking/<int:booking_id>/', views.accept_booking, name='accept_booking'),
    path('update-booking/<int:booking_id>/', views.update_booking_status, name='update_booking_status'),
    
    # History and Profile
    path('booking-history/', views.booking_history, name='booking_history'),
    path('notifications/', views.notifications, name='notifications'),
    path('profile/settings/', views.profile_settings, name='profile_settings'),
    
    # Chat Interface
    path('chat/', views.chat_interface, name='chat_interface'),
    
    # API endpoints
    path('api/submit-review/', views.submit_review_api, name='submit_review_api'),
    path('api/cancel-booking/<int:booking_id>/', views.cancel_booking_api, name='cancel_booking_api'),
    path('api/notifications/', views.notifications_api, name='notifications_api'),
    path('api/notifications/mark-read/', views.notifications_api, name='mark_notification_read'),
    path('api/notifications/mark-all-read/', views.notifications_api, name='mark_all_notifications_read'),
    path('api/notifications/delete/<int:notification_id>/', views.notifications_api, name='delete_notification'),
    path('api/notifications/clear-all/', views.notifications_api, name='clear_all_notifications'),
    path('api/notifications/unread-count/', views.notifications_api, name='unread_notifications_count'),
    path('api/chat/', views.chat_with_ai, name='chat_with_ai'),
    path('api/voice/', views.voice_assistant_api, name='voice_assistant'),
    path('api/crop-recommendations/', views.crop_recommendations_api, name='crop_recommendations'),
    path('api/update-profile/', views.update_profile_api, name='update_profile_api'),
    path('api/update-profile-picture/', views.update_profile_picture_api, name='update_profile_picture'),
    path('api/download-invoice/<int:booking_id>/', views.download_invoice_api, name='download_invoice'),
    path('api/nearby-workers/', views.api_nearby_workers, name='api_nearby_workers'),
    path('api/text-to-speech/', views.get_text_to_speech, name='get_text_to_speech'),
    path('api/language-switch/', views.language_switch, name='language_switch'),
]