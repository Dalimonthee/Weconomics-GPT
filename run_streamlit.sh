#!/bin/bash
echo "Starting Blockchain Knowledge Assistant..."
source venv/bin/activate
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false 