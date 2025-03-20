# Blockchain Knowledge Assistant

A Streamlit-powered interface for the Agentic RAG System that answers questions about blockchain technology.

## Features

- 💬 Ask any blockchain-related questions
- 🔍 View the intelligent search strategy 
- 📊 Examine search quality metrics
- 📚 Explore source documents used for answers
- ⚙️ Configure retrieval parameters

## Local Usage

1. Ensure you have all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure your `.env` file is set up with the necessary credentials:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   GOOGLE_API_KEY=your_google_gemini_api_key
   EMBEDDING_DIMENSION=768
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open your browser at http://localhost:8501

## Deployment Options

### Streamlit Community Cloud (Recommended)

1. Push your code to a GitHub repository

2. Visit [Streamlit Community Cloud](https://streamlit.io/cloud)

3. Sign in with your GitHub account

4. Click "New app" and select your repository, branch, and the `streamlit_app.py` file

5. Add your environment variables in the "Advanced settings" section

6. Deploy the app!

### Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)

2. Choose Streamlit as the SDK

3. Upload your code to the Space using Git or the Web UI

4. Set up your environment variables in the "Settings" tab

5. The app will automatically deploy

### Render

1. Create a new Web Service on [Render](https://render.com)

2. Connect your GitHub repository

3. Set the build command to `pip install -r requirements.txt`

4. Set the start command to `streamlit run streamlit_app.py`

5. Add your environment variables

6. Deploy the app

## Tips for Demo

For the best demonstration experience:

1. Start with simple questions like "What is blockchain?" or "How does a consensus mechanism work?"

2. Show how the system intelligently reformulates queries and identifies key concepts

3. Use the "Database Statistics" button to show the size of the knowledge base

4. Adjust the "Minimum content length" slider to show how it affects search quality

5. Toggle "Verbose mode" to show or hide detailed search information

## Limitations

- The system is limited to the blockchain domain knowledge
- Response quality depends on the available content in the Supabase vector database
- Processing time may vary based on query complexity and connection speed 