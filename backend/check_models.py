import google.genai as genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment.")
else:
    client = genai.Client(api_key=api_key)
    print("Available Gemini models (supports generateContent):")
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            print(f"  {m.name}")
