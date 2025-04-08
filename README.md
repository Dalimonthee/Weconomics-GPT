# Weconomics AI Chat

A Django application that integrates with your college's AI system, providing a user-friendly chat interface with persistent conversation history.

## Features

- User authentication with Django's built-in authentication system
- Conversation management with persistent storage
- Integration with external AI services
- Responsive web interface for asking questions and viewing answers
- Admin interface for managing users and their conversations

## Setup for Local Development

1. Clone the repository:
   ```
   git clone <repository-url>
   cd weconomics-ai
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run migrations to set up the database:
   ```
   python manage.py migrate
   ```

4. Create a superuser for admin access:
   ```
   python manage.py createsuperuser
   ```

5. Run the development server:
   ```
   python manage.py runserver
   ```

6. Visit http://127.0.0.1:8000/ in your browser.

## Deployment on Render

### Prerequisites
- A Render account (https://render.com/)
- A GitHub repository with your Django application

### Steps

1. Log in to your Render account and create a new PostgreSQL database:
   - Go to "New" > "PostgreSQL"
   - Configure name, region, etc.
   - Click "Create Database"
   - Copy the "Internal Database URL" for the next step

2. Create a new Web Service:
   - Go to "New" > "Web Service"
   - Connect to your GitHub repository
   - Configure the service:
     - Name: weconomics-ai
     - Environment: Python 3
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn weconomics_ai.wsgi:application`

3. Add Environment Variables:
   - `DATABASE_URL`: Paste the PostgreSQL URL from step 1
   - `DJANGO_ENV`: production
   - `DJANGO_SECRET_KEY`: A secure random string
   - `ALLOWED_HOSTS`: Your Render domain, e.g., `yourdomain.onrender.com`

4. Deploy the service.

5. After the first deployment, run migrations:
   - Go to "Shell" in your web service dashboard
   - Run: `python manage.py migrate`
   - Create a superuser: `python manage.py createsuperuser`

6. Your application is now live at the URL provided by Render.

## AI Integration

The current implementation includes a placeholder function `call_ai_service()` in `chat/views.py`, which should be replaced with the actual integration code for your college's AI system.

To integrate with your college's AI system:

1. Understand the API contract required by your AI system
2. Update the `call_ai_service()` function to make the appropriate API calls
3. Handle any authentication or security requirements
4. Process the response from the AI system and return it as a string

## License

[Your License Information] 