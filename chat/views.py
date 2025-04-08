from django.shortcuts import render, redirect, get_object_or_404
import json
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from .models import Conversation, ChatMessage
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from django.views.generic.edit import CreateView
from django.urls import reverse_lazy
from django.contrib import messages
from .forms import CustomUserCreationForm
from django.utils import timezone
from .utils import query_knowledge_base
from django.contrib.auth import update_session_auth_hash

# AI service integration with RAG
def call_ai_service(question, context=None):
    """
    Call the AI service to get an answer for the user's question.
    
    This is a placeholder function that would be replaced with actual integration
    to your college's AI system.
    
    Args:
        question: The user's question
        context: Optional conversation context
        
    Returns:
        str: The AI's response
    """
    try:
        # Step 1: Query the knowledge base to retrieve relevant documents
        relevant_documents = query_knowledge_base(question, context)
        
        # Step 2: Format the context for the AI service
        knowledge_context = ""
        if relevant_documents:
            knowledge_context = "Based on the following information:\n\n"
            for i, doc in enumerate(relevant_documents, 1):
                knowledge_context += f"{i}. {doc['content']}\n\n"
            knowledge_context += "Answer the question: " + question
        
        # Step 3: In a real implementation, this would call the external AI service
        # with the question and knowledge context
        
        # For now, return a placeholder response that demonstrates the RAG process
        if not relevant_documents:
            return f"I don't have specific information to answer that question. Here's a general response: This is a placeholder answer from the AI service for: '{question}'"
        
        return f"Based on the retrieved knowledge, here's an answer to your question: '{question}'\n\nThis is a placeholder response that would come from your college's AI system, using the relevant documents retrieved from the knowledge base."
        
    except Exception as e:
        # Log the error and return a friendly message
        print(f"Error calling AI service: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your question. Please try again later."

@login_required
def conversation_list(request):
    """Display a list of user's conversations."""
    conversations = Conversation.objects.filter(user=request.user).order_by('-updated_at')
    return render(request, 'chat/conversation_list.html', {'conversations': conversations})

@login_required
def conversation_detail(request, conversation_id):
    """Display a specific conversation with all messages."""
    conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
    messages = conversation.messages.all().order_by('timestamp')
    return render(request, 'chat/conversation_detail.html', {
        'conversation': conversation,
        'messages': messages
    })

@login_required
def new_conversation(request):
    """Start a new conversation."""
    conversation = Conversation.objects.create(user=request.user)
    return redirect('conversation_detail', conversation_id=conversation.id)

@csrf_exempt
@require_POST
@login_required
def ask_question(request):
    """Handle a user's question to the AI service."""
    data = json.loads(request.body)
    question_text = data.get("question")
    conv_id = data.get("conversation_id")
    
    # Retrieve or create conversation
    if conv_id:
        try:
            conversation = Conversation.objects.get(id=conv_id, user=request.user)
        except Conversation.DoesNotExist:
            return JsonResponse({"error": "Conversation not found."}, status=404)
    else:
        conversation = Conversation.objects.create(user=request.user)

    # Save user's question
    user_message = ChatMessage.objects.create(
        conversation=conversation,
        sender='user',
        message=question_text
    )
    
    # Optional: retrieve previous messages as context (e.g., last few messages)
    context_messages = conversation.messages.order_by('-timestamp')[:5]
    context = "\n".join([msg.message for msg in context_messages])

    # Call the AI service to get an answer
    ai_answer = call_ai_service(question_text, context=context)
    
    # Save AI's response
    ai_message = ChatMessage.objects.create(
        conversation=conversation,
        sender='ai',
        message=ai_answer
    )
    
    # Return the conversation update (question and answer)
    response_data = {
        "conversation_id": conversation.id,
        "messages": [
            {"sender": user_message.sender, "message": user_message.message, "timestamp": user_message.timestamp.isoformat()},
            {"sender": ai_message.sender, "message": ai_message.message, "timestamp": ai_message.timestamp.isoformat()}
        ]
    }
    return JsonResponse(response_data)

class RegisterView(CreateView):
    form_class = CustomUserCreationForm
    template_name = 'auth/register.html'
    success_url = reverse_lazy('login')
    
    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, "Your account has been created successfully. You can now log in.")
        return response

@login_required
def profile_view(request):
    """User profile page"""
    if request.method == 'POST':
        # Handle profile updates
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        
        user = request.user
        user.first_name = first_name
        user.last_name = last_name
        user.email = email
        user.save()
        
        messages.success(request, "Your profile has been updated successfully.")
        return redirect('profile')
        
    return render(request, 'auth/profile.html', {
        'user': request.user
    })

@login_required
def rename_conversation(request, conversation_id):
    """Rename a conversation"""
    conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
    
    if request.method == 'POST':
        new_title = request.POST.get('title', '').strip()
        if new_title:
            conversation.title = new_title
            conversation.save()
            messages.success(request, "Conversation renamed successfully.")
        else:
            messages.error(request, "Title cannot be empty.")
        
        return redirect('conversation_detail', conversation_id=conversation.id)
    
    return render(request, 'chat/rename_conversation.html', {
        'conversation': conversation
    })

@login_required
def delete_conversation(request, conversation_id):
    """Delete a conversation"""
    conversation = get_object_or_404(Conversation, id=conversation_id, user=request.user)
    
    if request.method == 'POST':
        conversation.delete()
        messages.success(request, "Conversation deleted successfully.")
        return redirect('conversation_list')
    
    return render(request, 'chat/delete_conversation.html', {
        'conversation': conversation
    })

@login_required
def change_password(request):
    """Change user password"""
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            # Update the session to prevent user from being logged out
            update_session_auth_hash(request, user)
            messages.success(request, "Your password has been changed successfully!")
            return redirect('profile')
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = PasswordChangeForm(request.user)
    
    return render(request, 'auth/change_password.html', {
        'form': form
    })
