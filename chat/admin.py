from django.contrib import admin
from django.urls import path
from .models import Source, SourceDocument
from .admin_views import source_management, process_source
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User

# Removed Conversation and ChatMessage to protect user privacy

# Custom User Admin with enhanced management features
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'date_joined', 'last_login')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'date_joined')
    search_fields = ('username', 'email', 'first_name', 'last_name')
    readonly_fields = ('date_joined', 'last_login')
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'email')}),
        ('Permissions', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions'),
            'classes': ('collapse',),
        }),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )
    actions = ['reset_password']
    
    def reset_password(self, request, queryset):
        # Set a temporary password for selected users
        temp_password = 'temp123'  # In a real app, generate a random password
        for user in queryset:
            user.set_password(temp_password)
            user.save()
        self.message_user(request, f"{queryset.count()} users have had their passwords reset.")
    reset_password.short_description = "Reset password for selected users"

# Re-register the User model with our custom admin
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)

class SourceDocumentInline(admin.TabularInline):
    model = SourceDocument
    extra = 1
    fields = ('title', 'file', 'url', 'is_processed', 'processing_status')
    readonly_fields = ('is_processed', 'processing_status')

@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    list_display = ('name', 'source_type', 'created_by', 'created_at', 'is_active')
    list_filter = ('source_type', 'is_active', 'created_at')
    search_fields = ('name', 'description')
    inlines = [SourceDocumentInline]
    
    def save_model(self, request, obj, form, change):
        if not change:  # If this is a new object
            obj.created_by = request.user
        super().save_model(request, obj, form, change)
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('source-management/', self.admin_site.admin_view(source_management), name='source_management'),
            path('process-source/<int:source_id>/', self.admin_site.admin_view(process_source), name='process_source'),
        ]
        return custom_urls + urls

@admin.register(SourceDocument)
class SourceDocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'source', 'is_processed', 'processing_status', 'created_at')
    list_filter = ('source', 'is_processed', 'processing_status', 'created_at')
    search_fields = ('title', 'content', 'source__name')
    readonly_fields = ('last_processed_at',)
    fieldsets = (
        (None, {
            'fields': ('source', 'title')
        }),
        ('Content', {
            'fields': ('content', 'file', 'url'),
            'classes': ('wide',)
        }),
        ('Processing Status', {
            'fields': ('is_processed', 'processing_status', 'last_processed_at'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('metadata',),
            'classes': ('collapse',)
        }),
    )
