from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import create_app

app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
