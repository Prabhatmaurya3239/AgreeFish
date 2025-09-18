from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_http_methods
from django.utils import timezone
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.paginator import Paginator
from django.db.models import Q, Avg, Count
from datetime import datetime, timedelta
import json
import razorpay
import requests
from .models import *
from .forms import *
from .utils import *
from .utils import detect_plant_disease_real

def home(request):
    """Home page with role selection"""
    return render(request, 'home.html')

def register_farmer(request):
    """Farmer registration"""
    if request.method == 'POST':
        form = FarmerRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            # Create user
            user = User.objects.create_user(
                username=form.cleaned_data['username'],
                email=form.cleaned_data['email'],
                password=form.cleaned_data['password1'],
                first_name=form.cleaned_data['first_name'],
                last_name=form.cleaned_data['last_name']
            )
            
            # Create user profile
            user_profile = UserProfile.objects.create(
                user=user,
                user_type='farmer',
                phone=form.cleaned_data['phone'],
                address=form.cleaned_data['address'],
                latitude=form.cleaned_data.get('latitude'),
                longitude=form.cleaned_data.get('longitude'),
                preferred_language=form.cleaned_data['preferred_language']
            )
            
            # Create farmer profile
            FarmerProfile.objects.create(
                user_profile=user_profile,
                farm_name=form.cleaned_data['farm_name'],
                farm_size_acres=form.cleaned_data['farm_size_acres'],
                crops_grown=form.cleaned_data['crops_grown']
            )
            
            login(request, user)
            messages.success(request, 'Registration successful! Welcome to AgriFish!')
            return redirect('farmer_dashboard')
    else:
        form = FarmerRegistrationForm()
    
    return render(request, 'registration/register_farmer.html', {'form': form})

def register_worker(request):
    """Worker registration"""
    if request.method == 'POST':
        form = WorkerRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            # Create user
            user = User.objects.create_user(
                username=form.cleaned_data['username'],
                email=form.cleaned_data['email'],
                password=form.cleaned_data['password1'],
                first_name=form.cleaned_data['first_name'],
                last_name=form.cleaned_data['last_name']
            )
            
            # Create user profile
            user_profile = UserProfile.objects.create(
                user=user,
                user_type='worker',
                phone=form.cleaned_data['phone'],
                address=form.cleaned_data['address'],
                latitude=form.cleaned_data.get('latitude'),
                longitude=form.cleaned_data.get('longitude'),
                preferred_language=form.cleaned_data['preferred_language']
            )
            
            # Create worker profile
            WorkerProfile.objects.create(
                user_profile=user_profile,
                service_radius_km=form.cleaned_data['service_radius_km'],
                hourly_rate=form.cleaned_data['hourly_rate'],
                experience_years=form.cleaned_data['experience_years'],
                equipment_owned=form.cleaned_data['equipment_owned']
            )
            
            login(request, user)
            messages.success(request, 'Registration successful! Welcome to AgriFish!')
            return redirect('worker_dashboard')
    else:
        form = WorkerRegistrationForm()
    
    return render(request, 'registration/register_worker.html', {'form': form})

@login_required
def farmer_dashboard(request):
    """Farmer dashboard"""
    if not hasattr(request.user, 'userprofile') or request.user.userprofile.user_type != 'farmer':
        messages.error(request, 'Access denied. Farmer account required.')
        return redirect('home')
    
    farmer_profile = request.user.userprofile
    recent_detections = CropDiseaseDetection.objects.filter(farmer=farmer_profile)[:5]
    active_bookings = ServiceBooking.objects.filter(farmer=farmer_profile).exclude(status__in=['completed', 'cancelled'])[:3]
    notifications = Notification.objects.filter(user=farmer_profile, is_read=False)[:5]
    
    context = {
        'farmer_profile': farmer_profile,
        'recent_detections': recent_detections,
        'active_bookings': active_bookings,
        'notifications': notifications,
        'google_maps_key': settings.GOOGLE_MAPS_API_KEY,
    }
    return render(request, 'farmer/dashboard.html', context)

@login_required
def worker_dashboard(request):
    """Enhanced worker dashboard with reviews"""
    if not hasattr(request.user, 'userprofile') or request.user.userprofile.user_type != 'worker':
        messages.error(request, 'Access denied. Worker account required.')
        return redirect('home')
    
    worker_profile = request.user.userprofile
    
    # Get nearby pending bookings
    pending_bookings = ServiceBooking.objects.filter(
        worker=None,
        status='pending',
        payment_status='paid'
    ).order_by('-created_at')[:10]
    
    # Filter by location if worker has coordinates
    if worker_profile.latitude and worker_profile.longitude:
        radius = worker_profile.workerprofile.service_radius_km
        # Add distance filtering logic here
        nearby_bookings = []
        for booking in pending_bookings:
            if booking.farm_latitude and booking.farm_longitude:
                distance = calculate_distance(
                    worker_profile.latitude, worker_profile.longitude,
                    booking.farm_latitude, booking.farm_longitude
                )
                if distance <= radius:
                    nearby_bookings.append(booking)
        pending_bookings = nearby_bookings[:10]
    
    my_bookings = ServiceBooking.objects.filter(
        worker=worker_profile
    ).order_by('-created_at')[:10]
    
    notifications = Notification.objects.filter(
        user=worker_profile, 
        is_read=False
    )[:5]
    
    # Get recent reviews
    recent_reviews = WorkerReview.objects.filter(
        worker=worker_profile
    ).order_by('-created_at')[:5]
    
    context = {
        'worker_profile': worker_profile,
        'pending_bookings': pending_bookings,
        'my_bookings': my_bookings,
        'notifications': notifications,
        'recent_reviews': recent_reviews,
        'avg_rating': worker_profile.workerprofile.rating,
        'total_jobs': worker_profile.workerprofile.total_jobs,
    }
    return render(request, 'worker/dashboard.html', context)
@login_required
def disease_detection(request):
    """Enhanced disease detection with REAL working AI"""
    if not hasattr(request.user, 'userprofile') or request.user.userprofile.user_type != 'farmer':
        messages.error(request, 'Access denied. Farmer account required.')
        return redirect('home')
    
    if request.method == 'POST':
        form = DiseaseDetectionForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Save detection record first
                detection = form.save(commit=False)
                detection.farmer = request.user.userprofile
                detection.save()
                
                print(f"Processing disease detection for user: {request.user.username}")
                print(f"Image path: {detection.crop_image.path}")
                print(f"Crop type: {detection.crop_type}")
                
                # Get user's preferred language
                user_language = request.user.userprofile.preferred_language
                
                # Process image with REAL AI detection
                result = detect_plant_disease_real(
                    image_path=detection.crop_image.path,
                    crop_type=detection.crop_type,
                    user_language=user_language
                )
                
                print(f"Detection result: {result}")
                
                if result.get('success', False):
                    # Update detection with AI results
                    detection.disease_detected = result.get('disease_detected', 'Unknown')
                    detection.confidence_score = result.get('confidence_score', 0.0)
                    detection.status = 'diseased' if not result.get('is_healthy', True) else 'healthy'
                    detection.suggestions = result.get('treatment', 'No specific treatment needed.')
                    
                    # Store full AI analysis
                    detection.ai_analysis = result
                    detection.plant_id = result.get('plant_id', '')
                    detection.disease_id = result.get('disease_id', '')
                    detection.severity = result.get('severity', 'medium')
                    detection.urgency = result.get('urgency', 'routine')
                    detection.analysis_method = result.get('method', 'ai_detection')
                    
                    # Store translated suggestions
                    translated_suggestions = {}
                    if result.get('treatment_translated'):
                        translated_suggestions[user_language] = result['treatment_translated']
                    if result.get('prevention_translated'):
                        translated_suggestions[f'{user_language}_prevention'] = result['prevention_translated']
                    
                    detection.suggestions_translated = translated_suggestions
                    detection.save()
                    
                    # Create notification
                    disease_status = "Disease Detected" if detection.disease_detected else "Healthy Crop"
                    notification_message = f'Analysis complete for {detection.crop_type}. {disease_status}: {detection.disease_detected or "No disease found"}.'
                    
                    Notification.objects.create(
                        user=request.user.userprofile,
                        title='Disease Analysis Complete',
                        message=notification_message,
                        notification_type='disease_detected',
                        related_detection=detection
                    )
                    
                    success_message = f"Disease detection completed! {disease_status}"
                    if detection.confidence_score:
                        success_message += f" (Confidence: {detection.confidence_score:.1f}%)"
                    
                    messages.success(request, success_message)
                    return redirect('detection_result', detection_id=detection.id)
                
                else:
                    # AI detection failed
                    error_msg = result.get('error', 'Unknown error occurred')
                    detection.status = 'unknown'
                    detection.suggestions = f'Analysis failed: {error_msg}. Please try uploading a clearer image.'
                    detection.ai_analysis = result
                    detection.save()
                    
                    messages.error(request, f'Disease detection failed: {error_msg}')
                    return redirect('detection_result', detection_id=detection.id)
                
            except Exception as e:
                print(f"Disease detection error: {e}")
                
                # Update detection with error info
                if 'detection' in locals():
                    detection.status = 'unknown'
                    detection.suggestions = f'Processing error: {str(e)}. Please try again.'
                    detection.save()
                    messages.error(request, f'Error processing image: {str(e)}')
                    return redirect('detection_result', detection_id=detection.id)
                else:
                    messages.error(request, f'Error saving image: {str(e)}')
    else:
        form = DiseaseDetectionForm()
    
    # Get recent detections for reference
    recent_detections = CropDiseaseDetection.objects.filter(
        farmer=request.user.userprofile
    ).order_by('-detected_at')[:5]
    
    context = {
        'form': form,
        'recent_detections': recent_detections,
        'supported_crops': [
            'tomato', 'potato', 'wheat', 'rice', 'corn', 
            'cotton', 'soybean', 'apple', 'grape'
        ]
    }
    
    return render(request, 'farmer/disease_detection.html', context)

@login_required
def detection_result(request, detection_id):
    """Enhanced detection result with working voice and recommendations"""
    detection = get_object_or_404(
        CropDiseaseDetection, 
        id=detection_id, 
        farmer=request.user.userprofile
    )
    
    # Get AI analysis result
    ai_result = detection.ai_analysis or {}
    
    # Get user's preferred language
    user_language = request.user.userprofile.preferred_language
    
    # Get translated content
    treatment_text = detection.suggestions
    prevention_text = ai_result.get('prevention', '')
    
    if detection.suggestions_translated:
        treatment_text = detection.suggestions_translated.get(
            user_language, 
            treatment_text
        )
        prevention_text = detection.suggestions_translated.get(
            f'{user_language}_prevention', 
            prevention_text
        )
    
    # Determine if service booking is recommended
    can_book_service = (
        detection.status == 'diseased' or 
        ai_result.get('severity') in ['medium', 'high'] or
        ai_result.get('urgency') in ['within_week', 'immediate']
    )
    
    # Get service urgency
    urgency_level = ai_result.get('urgency', 'routine')
    urgency_colors = {
        'immediate': 'danger',
        'within_week': 'warning', 
        'routine': 'info'
    }
    
    context = {
        'detection': detection,
        'ai_result': ai_result,
        'treatment_text': treatment_text,
        'prevention_text': prevention_text,
        'can_book_service': can_book_service,
        'severity': ai_result.get('severity', 'medium'),
        'urgency': urgency_level,
        'urgency_color': urgency_colors.get(urgency_level, 'info'),
        'confidence_percentage': f"{detection.confidence_score:.1f}%" if detection.confidence_score else "N/A",
        'symptoms': ai_result.get('symptoms_translated', ai_result.get('symptoms', [])),
        'analysis_method': ai_result.get('method', 'AI Analysis'),
        'user_language': user_language
    }
    
    return render(request, 'farmer/detection_result.html', context)

# API endpoint for testing detection
@login_required
def test_disease_detection_api(request):
    """API endpoint to test disease detection with uploaded image"""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            crop_type = request.POST.get('crop_type', 'tomato')
            
            # Save temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                for chunk in image_file.chunks():
                    tmp_file.write(chunk)
                tmp_path = tmp_file.name
            
            try:
                # Test detection
                result = detect_plant_disease_real(
                    tmp_path, 
                    crop_type, 
                    request.user.userprofile.preferred_language
                )
                
                return JsonResponse(result)
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False, 
        'error': 'No image provided'
    })
@login_required
def book_service(request, detection_id=None):
    """Book spray service"""
    detection = None
    if detection_id:
        detection = get_object_or_404(CropDiseaseDetection, id=detection_id, farmer=request.user.userprofile)
    
    if request.method == 'POST':
        form = ServiceBookingForm(request.POST)
        if form.is_valid():
            booking = form.save(commit=False)
            booking.farmer = request.user.userprofile
            booking.disease_detection = detection
            
            # Calculate estimated cost
            booking.estimated_cost = calculate_service_cost(
                booking.area_acres,
                booking.crop_type,
                detection.disease_detected if detection else None
            )
            
            booking.save()
            
            # Create notification
            Notification.objects.create(
                user=request.user.userprofile,
                title='Service Booking Created',
                message=f'Your service booking #{booking.booking_id} has been created.',
                notification_type='booking_created',
                related_booking=booking
            )
            
            messages.success(request, 'Service booking created successfully!')
            return redirect('booking_payment', booking_id=booking.id)
    else:
        initial_data = {}
        if detection:
            initial_data.update({
                'crop_type': detection.crop_type,
                'farm_latitude': request.user.userprofile.latitude,
                'farm_longitude': request.user.userprofile.longitude,
                'farm_address': request.user.userprofile.address,
            })
        form = ServiceBookingForm(initial=initial_data)
    
    context = {
        'form': form,
        'detection': detection,
        'google_maps_key': settings.GOOGLE_MAPS_API_KEY,
    }
    return render(request, 'farmer/book_service.html', context)

@login_required
def booking_payment(request, booking_id):
    """Payment page for booking"""
    booking = get_object_or_404(ServiceBooking, id=booking_id, farmer=request.user.userprofile)
    
    if request.method == 'POST':
        try:
            # Create Razorpay order
            client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))
            
            amount = int(booking.estimated_cost * 100)  # Convert to paise
            order_data = {
                'amount': amount,
                'currency': 'INR',
                'receipt': str(booking.booking_id),
                'notes': {
                    'booking_id': str(booking.booking_id),
                    'farmer': booking.farmer.user.username
                }
            }
            gst = booking.estimated_cost * 0.18
            total = booking.estimated_cost + gst
            order = client.order.create(data=order_data)
            booking.razorpay_order_id = order['id']
            booking.save()
            
            context = {
                'booking': booking,
                'order': order,
                'razorpay_key_id': settings.RAZORPAY_KEY_ID,
                'amount': amount,
                 "gst": gst,
                 "total": total,
            }
            return render(request, 'farmer/payment.html', context)
            
        except Exception as e:
            messages.error(request, f'Payment initialization failed: {str(e)}')
    
    return render(request, 'farmer/booking_payment.html', {'booking': booking ,  'razorpay_key_id' : settings.RAZORPAY_KEY_ID, })

@csrf_exempt
@require_POST
def payment_success(request):
    """Handle successful payment"""
    try:
        payment_id = request.POST.get('razorpay_payment_id')
        order_id = request.POST.get('razorpay_order_id')
        signature = request.POST.get('razorpay_signature')
        
        # Verify payment signature
        client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))
        
        params_dict = {
            'razorpay_order_id': order_id,
            'razorpay_payment_id': payment_id,
            'razorpay_signature': signature
        }
        
        client.utility.verify_payment_signature(params_dict)
        
        # Update booking
        booking = ServiceBooking.objects.get(razorpay_order_id=order_id)
        booking.payment_status = 'paid'
        booking.razorpay_payment_id = payment_id
        booking.save()
        
        # Create notification
        Notification.objects.create(
            user=booking.farmer,
            title='Payment Successful',
            message=f'Payment for booking #{booking.booking_id} completed successfully.',
            notification_type='payment_success',
            related_booking=booking
        )
        
        messages.success(request, 'Payment successful! Your booking is now active.')
        return redirect('farmer_dashboard')
        
    except Exception as e:
        messages.error(request, f'Payment verification failed: {str(e)}')
        return redirect('farmer_dashboard')

@login_required
def accept_booking(request, booking_id):
    """Worker accepts a booking"""
    booking = get_object_or_404(ServiceBooking, id=booking_id, status='pending')
    
    if request.user.userprofile.user_type != 'worker':
        messages.error(request, 'Only workers can accept bookings.')
        return redirect('home')
    
    booking.worker = request.user.userprofile
    booking.status = 'accepted'
    booking.accepted_at = timezone.now()
    booking.save()
    
    # Create notification for farmer
    Notification.objects.create(
        user=booking.farmer,
        title='Booking Accepted',
        message=f'Your booking #{booking.booking_id} has been accepted by {booking.worker.user.get_full_name()}.',
        notification_type='booking_accepted',
        related_booking=booking
    )
    
    messages.success(request, 'Booking accepted successfully!')
    return redirect('worker_dashboard')

@login_required
def update_booking_status(request, booking_id):
    """Update booking status by worker"""
    booking = get_object_or_404(ServiceBooking, id=booking_id, worker=request.user.userprofile)
    
    if request.method == 'POST':
        new_status = request.POST.get('status')
        notes = request.POST.get('notes', '')
        
        if new_status in ['in_progress', 'completed']:
            booking.status = new_status
            if new_status == 'completed':
                booking.completed_at = timezone.now()
            booking.save()
            
            # Create status update record
            BookingStatusUpdate.objects.create(
                booking=booking,
                status=new_status,
                notes=notes,
                updated_by=request.user.userprofile
            )
            
            # Create notification for farmer
            Notification.objects.create(
                user=booking.farmer,
                title=f'Booking {new_status.title()}',
                message=f'Your booking #{booking.booking_id} is now {new_status.replace("_", " ")}.',
                notification_type='booking_completed' if new_status == 'completed' else 'booking_accepted',
                related_booking=booking
            )
            
            messages.success(request, f'Booking status updated to {new_status.replace("_", " ")}.')
    
    return redirect('worker_dashboard')

@login_required
def booking_history(request):
    """Enhanced booking history with filters and pagination"""
    user_profile = request.user.userprofile
    
    # Base query
    if user_profile.user_type == 'farmer':
        bookings = ServiceBooking.objects.filter(farmer=user_profile)
        template = 'farmer/booking_history.html'
    else:
        bookings = ServiceBooking.objects.filter(worker=user_profile)
        template = 'worker/booking_history.html'
    
    # Apply filters
    status_filter = request.GET.get('status')
    if status_filter:
        bookings = bookings.filter(status=status_filter)
    
    payment_filter = request.GET.get('payment_status')
    if payment_filter:
        bookings = bookings.filter(payment_status=payment_filter)
    
    date_from = request.GET.get('date_from')
    if date_from:
        bookings = bookings.filter(created_at__gte=date_from)
    
    date_to = request.GET.get('date_to')
    if date_to:
        bookings = bookings.filter(created_at__lte=date_to)
    
    # Search
    search = request.GET.get('search')
    if search:
        bookings = bookings.filter(
            Q(crop_type__icontains=search) |
            Q(service_type__icontains=search) |
            Q(farm_address__icontains=search)
        )
    
    # Order by latest first
    bookings = bookings.order_by('-created_at')
    
    # Pagination
    paginator = Paginator(bookings, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'bookings': page_obj,
        'page_obj': page_obj,
        'is_paginated': page_obj.has_other_pages(),
        'total_bookings': bookings.count(),
        'completed_bookings': bookings.filter(status='completed').count(),
        'pending_bookings' : bookings.filter(status='pending').count(),
        'total_spent': sum([b.estimated_cost for b in bookings.filter(payment_status='paid')]),
    }
    
    return render(request, template, context)


@login_required
def get_text_to_speech(request):
    """Generate text-to-speech for suggestions"""
    text = request.GET.get('text', '')
    language = request.GET.get('language', 'en')
    
    try:
        audio_file = generate_speech(text, language)
        return JsonResponse({'success': True, 'audio_url': audio_file})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
def notifications(request):
    """Enhanced notifications page with categories"""
    user_notifications = Notification.objects.filter(
        user=request.user.userprofile
    ).order_by('-created_at')
    
    # Mark notifications as read when viewed
    if request.method == 'POST':
        notification_id = request.POST.get('notification_id')
        if notification_id:
            notification = get_object_or_404(Notification, 
                                           id=notification_id, 
                                           user=request.user.userprofile)
            notification.is_read = True
            notification.save()
            return JsonResponse({'success': True})
    
    # Pagination
    paginator = Paginator(user_notifications, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'notifications': page_obj,
        'page_obj': page_obj,
        'unread_count': user_notifications.filter(is_read=False).count(),
    }
    
    return render(request, 'notifications.html', context)
@login_required
def profile_settings(request):
    """User profile settings"""
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.userprofile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile_settings')
    else:
        form = ProfileUpdateForm(instance=request.user.userprofile)
    
    return render(request, 'profile_settings.html', {'form': form})

def api_nearby_workers(request):
    """API to get nearby available workers"""
    latitude = float(request.GET.get('lat', 0))
    longitude = float(request.GET.get('lng', 0))
    radius_km = int(request.GET.get('radius', 20))
    
    workers = get_nearby_workers(latitude, longitude, radius_km)
    
    worker_data = []
    for worker in workers:
        worker_data.append({
            'id': worker.id,
            'name': worker.user_profile.user.get_full_name(),
            'rating': worker.rating,
            'hourly_rate': float(worker.hourly_rate),
            'experience_years': worker.experience_years,
            'total_jobs': worker.total_jobs,
            'distance': calculate_distance(
                latitude, longitude,
                worker.user_profile.latitude, worker.user_profile.longitude
            )
        })
    
    return JsonResponse({'workers': worker_data})

def language_switch(request):
    """Switch user language preference"""
    if request.method == 'POST':
        language = request.POST.get('language', 'en')
        if request.user.is_authenticated and hasattr(request.user, 'userprofile'):
            request.user.userprofile.preferred_language = language
            request.user.userprofile.save()
        
        # Set session language
        request.session['language'] = language
        return JsonResponse({'success': True})
    
    return JsonResponse({'success': False})

@login_required
@require_POST
def submit_review_api(request):
    """API endpoint to submit worker review"""
    try:
        booking_id = request.POST.get('booking_id')
        rating = int(request.POST.get('rating'))
        comment = request.POST.get('comment', '')
        
        booking = get_object_or_404(ServiceBooking, 
                                   id=booking_id, 
                                   farmer=request.user.userprofile,
                                   status='completed')
        
        if hasattr(booking, 'workerreview'):
            return JsonResponse({'success': False, 'error': 'Review already exists'})
        
        # Create review
        review = WorkerReview.objects.create(
            booking=booking,
            farmer=request.user.userprofile,
            worker=booking.worker,
            rating=rating,
            comment=comment
        )
        
        # Update worker's average rating
        worker_profile = booking.worker.workerprofile
        avg_rating = WorkerReview.objects.filter(
            worker=booking.worker
        ).aggregate(Avg('rating'))['rating__avg']
        
        worker_profile.rating = round(avg_rating, 1)
        worker_profile.save()
        
        return JsonResponse({
            'success': True, 
            'message': 'Review submitted successfully',
            'new_rating': worker_profile.rating
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_POST
def cancel_booking_api(request, booking_id):
    """API endpoint to cancel booking"""
    try:
        booking = get_object_or_404(ServiceBooking, 
                                   id=booking_id, 
                                   farmer=request.user.userprofile,
                                   status='pending')
        
        booking.status = 'cancelled'
        booking.save()
        
        # Create notification for worker if assigned
        if booking.worker:
            Notification.objects.create(
                user=booking.worker,
                title='Booking Cancelled',
                message=f'Booking #{booking.booking_id} has been cancelled by the farmer.',
                notification_type='booking_cancelled',
                related_booking=booking
            )
        
        return JsonResponse({
            'success': True, 
            'message': 'Booking cancelled successfully'
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
def notifications_api(request):
    """API for notification management"""
    if request.method == 'POST':
        # Handle FormData or JSON
        try:
            data = json.loads(request.body)
        except:
            data = request.POST

        action = data.get('action')

        if action == 'mark_read':
            notification_id = data.get('notification_id')
            try:
                notification = Notification.objects.get(
                    id=notification_id,
                    user=request.user.userprofile
                )
                notification.is_read = True
                notification.save()
                return JsonResponse({'success': True})
            except:
                return JsonResponse({'success': False, 'error': 'Notification not found'})

        elif action == 'mark_all_read':
            Notification.objects.filter(
                user=request.user.userprofile,
                is_read=False
            ).update(is_read=True)
            return JsonResponse({'success': True})

    elif request.method == 'DELETE':
        # Handle delete requests
        if '/clear-all/' in request.path:
            Notification.objects.filter(user=request.user.userprofile).delete()
            return JsonResponse({'success': True})
        else:
            try:
                notification_id = request.path.split('/')[-2]
                Notification.objects.filter(
                    id=notification_id,
                    user=request.user.userprofile
                ).delete()
                return JsonResponse({'success': True})
            except:
                return JsonResponse({'success': False, 'error': 'Invalid request'})

    # GET request - return unread count
    unread_count = Notification.objects.filter(
        user=request.user.userprofile,
        is_read=False
    ).count()

    return JsonResponse({'count': unread_count})

@login_required
@csrf_exempt
def chat_with_ai(request):
    """Gemini AI chat assistant for farmers"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            context = data.get('context', '')
            
            # Get user's preferred language
            language = request.user.userprofile.preferred_language
            
            # Get AI response using Gemini
            ai_response = gemini_chat_assistant(
                user_message=user_message,
                context=context,
                language=language
            )
            
            return JsonResponse({
                'success': True,
                'response': ai_response,
                'timestamp': timezone.now().isoformat()
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e),
                'response': get_fallback_chat_response(user_message, language)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def voice_assistant_api(request):
    """Generate voice response for text"""
    if request.method == 'POST':
        try:
            text = request.POST.get('text', '')
            language = request.POST.get('language', request.user.userprofile.preferred_language)
            
            voice_result = generate_voice_response(text, language)
            
            return JsonResponse(voice_result)
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def crop_recommendations_api(request):
    """Get crop recommendations based on location"""
    try:
        lat = float(request.GET.get('lat', 0))
        lng = float(request.GET.get('lng', 0))
        season = request.GET.get('season', None)
        
        recommendations = get_crop_recommendations(lat, lng, season)
        
        return JsonResponse({
            'success': True,
            'recommendations': recommendations,
            'location': {'lat': lat, 'lng': lng},
            'season': season
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'recommendations': get_default_crop_recommendations('rabi')
        })

@login_required
def update_profile_api(request):
    """Update user profile via API"""
    if request.method == 'POST':
        try:
            # Update basic user info
            request.user.first_name = request.POST.get('first_name', request.user.first_name)
            request.user.last_name = request.POST.get('last_name', request.user.last_name)
            request.user.email = request.POST.get('email', request.user.email)
            request.user.save()
            
            # Update profile
            profile = request.user.userprofile
            profile.phone = request.POST.get('phone', profile.phone)
            profile.address = request.POST.get('address', profile.address)
            profile.preferred_language = request.POST.get('preferred_language', profile.preferred_language)
            
            # Update location if provided
            if request.POST.get('latitude') and request.POST.get('longitude'):
                profile.latitude = float(request.POST.get('latitude'))
                profile.longitude = float(request.POST.get('longitude'))
            
            profile.save()
            
            # Update type-specific profiles
            if profile.user_type == 'farmer' and hasattr(profile, 'farmerprofile'):
                farmer_profile = profile.farmerprofile
                farmer_profile.farm_name = request.POST.get('farm_name', farmer_profile.farm_name)
                farmer_profile.farm_size_acres = float(request.POST.get('farm_size_acres', farmer_profile.farm_size_acres))
                farmer_profile.crops_grown = request.POST.get('crops_grown', farmer_profile.crops_grown)
                farmer_profile.save()
                
            elif profile.user_type == 'worker' and hasattr(profile, 'workerprofile'):
                worker_profile = profile.workerprofile
                worker_profile.hourly_rate = float(request.POST.get('hourly_rate', worker_profile.hourly_rate))
                worker_profile.service_radius_km = int(request.POST.get('service_radius_km', worker_profile.service_radius_km))
                worker_profile.equipment_owned = request.POST.get('equipment_owned', worker_profile.equipment_owned)
                worker_profile.is_available = request.POST.get('is_available') == 'on'
                worker_profile.save()
            
            return JsonResponse({'success': True, 'message': 'Profile updated successfully'})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def update_profile_picture_api(request):
    """Update profile picture via API"""
    if request.method == 'POST' and request.FILES.get('profile_image'):
        try:
            profile = request.user.userprofile
            profile.profile_image = request.FILES['profile_image']
            profile.save()
            
            return JsonResponse({
                'success': True,
                'message': 'Profile picture updated successfully',
                'image_url': profile.profile_image.url
            })
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'No image provided'})

@login_required
def download_invoice_api(request, booking_id):
    """Generate and download invoice PDF"""
    try:
        booking = get_object_or_404(ServiceBooking, 
                                   id=booking_id, 
                                   farmer=request.user.userprofile)
        
        # Simple HTML invoice (can be enhanced with proper PDF generation)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Invoice #{booking.booking_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #28a745; }}
                .details {{ margin: 30px 0; }}
                .table {{ width: 100%; border-collapse: collapse; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                .total {{ background-color: #f8f9fa; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AgriFish</h1>
                <h2>Service Invoice</h2>
            </div>
            
            <div class="details">
                <p><strong>Invoice #:</strong> {booking.booking_id}</p>
                <p><strong>Date:</strong> {booking.created_at.strftime('%B %d, %Y')}</p>
                <p><strong>Farmer:</strong> {booking.farmer.user.get_full_name()}</p>
                <p><strong>Worker:</strong> {booking.worker.user.get_full_name() if booking.worker else 'N/A'}</p>
            </div>
            
            <table class="table">
                <thead>
                    <tr>
                        <th>Service</th>
                        <th>Crop Type</th>
                        <th>Area (Acres)</th>
                        <th>Amount</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{booking.service_type}</td>
                        <td>{booking.crop_type.title()}</td>
                        <td>{booking.area_acres}</td>
                        <td>₹{booking.estimated_cost}</td>
                    </tr>
                    <tr class="total">
                        <td colspan="3">Total</td>
                        <td>₹{booking.estimated_cost}</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="details">
                <p><strong>Payment Status:</strong> {booking.get_payment_status_display()}</p>
                <p><strong>Service Status:</strong> {booking.get_status_display()}</p>
            </div>
        </body>
        </html>
        """
        
        response = HttpResponse(html_content, content_type='text/html')
        response['Content-Disposition'] = f'attachment; filename="invoice_{booking.booking_id}.html"'
        
        return response
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    
    
@login_required
def chat_interface(request):
    """AI Chat interface for farmers"""
    return render(request, 'farmer/chat.html', {
        'user_language': request.user.userprofile.preferred_language
    })
