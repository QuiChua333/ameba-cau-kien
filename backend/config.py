import os
from pathlib import Path
from dotenv import load_dotenv

# Read env from the PROJECT-ROOT .env (one level above backend/), e.g.
# d:\quihn\ameba1\source\.env — that's where GEMINI_MODEL_NAME / GOOGLE_API_KEY live.
# Root takes precedence; a backend/.env (or cwd .env) only fills anything missing.
_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ROOT_ENV)
load_dotenv(override=False)

# Google Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    # Warning: You should ensure this environment variable is set in your deployment environment
    print("WARNING: GOOGLE_API_KEY environment variable not set.")

# Model Configuration
# Flash tier for speed (the pipeline is designed around Flash latency; the
# foundation table, foundation/pit elevations are now extracted deterministically
# from the text layer, so a faster vision model is low-risk).
# Override in .env with GEMINI_MODEL_NAME=... to use a different model.
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
