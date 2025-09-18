from django import template

register = template.Library()

@register.filter
def completed_count(bookings):
    # Agar bookings Page object hai to object_list nikal lo
    if hasattr(bookings, 'object_list'):
        bookings = bookings.object_list
    return bookings.filter(status='completed').count()
@register.filter
def pending_count(bookings):
    if hasattr(bookings, 'object_list'):
        bookings = bookings.object_list
    return bookings.filter(status='pending').count()
@register.filter
def total_spent(bookings):
    return sum([b.estimated_cost for b in bookings.filter(payment_status='paid')])

@register.filter
def completed_count(bookings):
    if hasattr(bookings, 'object_list'):
        bookings = bookings.object_list
    return bookings.filter(status='completed').count()
