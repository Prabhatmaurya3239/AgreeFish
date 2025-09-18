from django import template
register = template.Library()

@register.filter
def completed_count(bookings):
    return bookings.filter(status='completed').count()

@register.filter
def pending_count(bookings):
    return bookings.filter(status__in=['pending', 'accepted', 'in_progress']).count()

@register.filter
def total_spent(bookings):
    return sum([b.estimated_cost for b in bookings.filter(payment_status='paid')])

@register.filter
def mul(value, factor):
    """Multiply filter for template calculations"""
    try:
        return float(value) * float(factor)
    except (ValueError, TypeError):
        return 0