"""
Route blueprints for RDB web interface.
"""

from .search import search_bp
from .api import api_bp

__all__ = ["search_bp", "api_bp"]
