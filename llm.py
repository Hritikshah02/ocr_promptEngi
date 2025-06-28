import json
import re
import google.generativeai as genai
from config import GOOGLE_API_KEY

# Configure the Gemini API with the API key
genai.configure(api_key=GOOGLE_API_KEY)

def call_gemini(prompt: str) -> dict:
    """
    Send prompt to Google Gemini and return parsed JSON.
    """
    # List available models to debug
    try:
        available_models = genai.list_models()
        model_names = [model.name for model in available_models]
        print(f"Available models: {model_names}")
    except Exception as e:
        print(f"Could not list models: {str(e)}")
    
    # Configure the model - use a lighter model with less quota requirements
    model = genai.GenerativeModel(
        model_name="models/gemini-1.5-flash",  # Using lighter model for free tier
        generation_config={
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
    )
    
    # Add retry logic for rate limiting
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Generate content
            response = model.generate_content(prompt)
            
            # Get the response text
            response_text = response.text
            
            # Try to parse as JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: extract JSON substring
                match = re.search(r"\{.*\}", response_text, re.S)
                if match:
                    return json.loads(match.group())
                raise ValueError(f"Could not extract JSON from response: {response_text[:100]}...")
                
        except Exception as e:
            print(f"Gemini API error (attempt {attempt+1}/{max_retries}): {str(e)}")
            
            # If this is the last attempt, raise the exception
            if attempt == max_retries - 1:
                raise
                
            # Otherwise, wait and retry
            import time
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            # Increase delay for next attempt
            retry_delay *= 2
