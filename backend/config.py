import os
from dotenv import load_dotenv

load_dotenv()

# Google Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    # Warning: You should ensure this environment variable is set in your deployment environment
    print("WARNING: GOOGLE_API_KEY environment variable not set.")

# Model Configuration
# Model Configuration
# Using gemini-2.0-flash-lite-preview-02-05 for maximum speed (lowest latency)
GEMINI_MODEL_NAME = "gemini-3.1-pro-preview"
