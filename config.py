import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google API key for Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found in environment variables. Set it in .env file.")

# Endpoint for Google Gemini API
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
