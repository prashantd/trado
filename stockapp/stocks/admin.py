from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, RetestStock, BullishStock, BearishStock, StockUpdateLog, Instrument

class CustomUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (
        ('Additional Info', {'fields': ('name', 'api_key')}),
    )
    list_display = ('username', 'email', 'name', 'api_key', 'is_staff', 'is_active')
    search_fields = ('username', 'email', 'name')

# Register your models here.
admin.site.register(User, CustomUserAdmin)
admin.site.register(RetestStock)
admin.site.register(BullishStock)
admin.site.register(BearishStock)
admin.site.register(StockUpdateLog)
admin.site.register(Instrument)
