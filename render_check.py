# render_check.py
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# tohle je přesně to, na čem Render padal:
from deploy_backend.web_service import app

print("✅ import OK, FastAPI app found.")
