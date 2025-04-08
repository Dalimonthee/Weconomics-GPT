from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=255, blank=True, default='New Conversation')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Conversation {self.id} for {self.user.username}"

class ChatMessage(models.Model):
    SENDER_CHOICES = (
        ('user', 'User'),
        ('ai', 'AI'),
    )
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    sender = models.CharField(max_length=10, choices=SENDER_CHOICES)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.sender.capitalize()} at {self.timestamp}"

class Source(models.Model):
    """A knowledge source that can be used in the RAG process"""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    source_type = models.CharField(max_length=50, choices=[
        ('document', 'Document'),
        ('website', 'Website'),
        ('database', 'Database'),
        ('api', 'API')
    ])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='created_sources')
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name

class SourceDocument(models.Model):
    """Individual documents within a source"""
    source = models.ForeignKey(Source, on_delete=models.CASCADE, related_name='documents')
    title = models.CharField(max_length=255)
    content = models.TextField()
    file = models.FileField(upload_to='source_documents/', blank=True, null=True)
    url = models.URLField(blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Fields for tracking RAG processing status
    is_processed = models.BooleanField(default=False)
    processing_status = models.CharField(max_length=50, default='pending', choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ])
    last_processed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.title} ({self.source.name})"
