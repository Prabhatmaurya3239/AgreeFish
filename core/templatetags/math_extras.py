from django import template

register = template.Library()

@register.filter
def mul(value, arg):
    """Multiply the value by the argument."""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return ''
@register.filter
def add_gst(value, percent):
    """Add GST percent to value"""
    try:
        return float(value) + (float(value) * float(percent))
    except (ValueError, TypeError):
        return value