from django.contrib import admin
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.db.models import Count
from django.urls import path
from .models import Source, SourceDocument
from .utils import process_document

@staff_member_required
def source_management(request):
    """Admin view for managing knowledge sources"""
    sources = Source.objects.all().prefetch_related('documents')
    
    # Count documents by processing status
    status_counts = {
        'pending': SourceDocument.objects.filter(processing_status='pending').count(),
        'processing': SourceDocument.objects.filter(processing_status='processing').count(),
        'completed': SourceDocument.objects.filter(processing_status='completed').count(),
        'failed': SourceDocument.objects.filter(processing_status='failed').count(),
    }
    
    return render(request, 'admin/source_management.html', {
        'sources': sources,
        'status_counts': status_counts,
        'title': 'Knowledge Source Management',
    })

@staff_member_required
def process_source(request, source_id):
    """Process all documents for a source"""
    source = get_object_or_404(Source, id=source_id)
    documents = source.documents.filter(is_processed=False)
    
    if not documents.exists():
        messages.info(request, f"No unprocessed documents found for source: {source.name}")
        return redirect('admin:source_management')
    
    # In a real app, this would be handled by a background task
    # For simplicity, we'll process them directly
    processed_count = 0
    for doc in documents:
        success = process_document(doc.id)
        if success:
            processed_count += 1
    
    if processed_count > 0:
        messages.success(request, f"Successfully processed {processed_count} document(s) for source: {source.name}")
    else:
        messages.error(request, f"Failed to process documents for source: {source.name}")
    
    return redirect('admin:source_management')

# This class provides URL patterns for the admin views
class SourceAdminViews:
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('source-management/', source_management, name='source_management'),
            path('process-source/<int:source_id>/', process_source, name='process_source'),
        ]
        return custom_urls + urls 