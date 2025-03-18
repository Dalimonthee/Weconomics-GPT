@echo off
echo Starting Blockchain Knowledge Assistant...
call venv\Scripts\activate.bat
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
pause 