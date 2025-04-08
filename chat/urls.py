from django.urls import path
from . import views

urlpatterns = [
    path('', views.conversation_list, name='conversation_list'),
    path('new/', views.new_conversation, name='new_conversation'),
    path('<int:conversation_id>/', views.conversation_detail, name='conversation_detail'),
    path('<int:conversation_id>/rename/', views.rename_conversation, name='rename_conversation'),
    path('<int:conversation_id>/delete/', views.delete_conversation, name='delete_conversation'),
    path('api/ask/', views.ask_question, name='ask_question'),
] 