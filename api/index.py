import os
import sys
from pathlib import Path

# Add the project root to the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load .env variables for local testing, Vercel will have them in its environment config
from dotenv import load_dotenv
load_dotenv()

from backend.main import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
