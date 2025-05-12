# app/__init__.py

__version__ = "0.1.0"
# Optional: re-export commonly used members
from .main import app  # so someone can import the FastAPI instance as `app.app`
