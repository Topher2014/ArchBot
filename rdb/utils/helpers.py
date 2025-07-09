"""
Helper utilities for RDB.
"""

import re
import hashlib
import time
from pathlib import Path
from typing import Union, List, Dict, Any
from datetime import datetime, timedelta


def sanitize_filename(filename: str, max_length: int = 200) -> str:
   """Sanitize a string to be safe for use as a filename."""
   # Remove or replace dangerous characters
   sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
   
   # Remove control characters
   sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
   
   # Collapse multiple underscores/spaces
   sanitized = re.sub(r'[_\s]+', '_', sanitized)
   
   # Remove leading/trailing underscores and dots
   sanitized = sanitized.strip('_. ')
   
   # Truncate if too long
   if len(sanitized) > max_length:
       sanitized = sanitized[:max_length].rstrip('_. ')
   
   # Ensure it's not empty
   if not sanitized:
       sanitized = "unnamed"
   
   return sanitized


def calculate_hash(content: Union[str, bytes], algorithm: str = 'md5') -> str:
   """Calculate hash of content."""
   if isinstance(content, str):
       content = content.encode('utf-8')
   
   if algorithm == 'md5':
       return hashlib.md5(content).hexdigest()
   elif algorithm == 'sha1':
       return hashlib.sha1(content).hexdigest()
   elif algorithm == 'sha256':
       return hashlib.sha256(content).hexdigest()
   else:
       raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def format_bytes(bytes_count: int) -> str:
   """Format byte count as human-readable string."""
   for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
       if bytes_count < 1024.0:
           return f"{bytes_count:.1f} {unit}"
       bytes_count /= 1024.0
   return f"{bytes_count:.1f} PB"


def format_duration(seconds: float) -> str:
   """Format duration in seconds as human-readable string."""
   if seconds < 1:
       return f"{seconds*1000:.0f}ms"
   elif seconds < 60:
       return f"{seconds:.1f}s"
   elif seconds < 3600:
       minutes = int(seconds // 60)
       secs = int(seconds % 60)
       return f"{minutes}m {secs}s"
   else:
       hours = int(seconds // 3600)
       minutes = int((seconds % 3600) // 60)
       return f"{hours}h {minutes}m"


def chunk_list(lst: List, chunk_size: int) -> List[List]:
   """Split a list into chunks of specified size."""
   return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
   """Flatten a nested dictionary."""
   items = []
   for k, v in d.items():
       new_key = f"{parent_key}{sep}{k}" if parent_key else k
       if isinstance(v, dict):
           items.extend(flatten_dict(v, new_key, sep=sep).items())
       else:
           items.append((new_key, v))
   return dict(items)


def ensure_directory(path: Union[str, Path]) -> Path:
   """Ensure directory exists, create if it doesn't."""
   path = Path(path)
   path.mkdir(parents=True, exist_ok=True)
   return path


def get_file_age(file_path: Union[str, Path]) -> timedelta:
   """Get age of a file."""
   file_path = Path(file_path)
   if not file_path.exists():
       raise FileNotFoundError(f"File not found: {file_path}")
   
   file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
   return datetime.now() - file_time


def retry_on_exception(max_retries: int = 3, delay: float = 1.0, 
                     backoff: float = 2.0, exceptions: tuple = (Exception,)):
   """Decorator to retry function calls on specified exceptions."""
   def decorator(func):
       def wrapper(*args, **kwargs):
           current_delay = delay
           last_exception = None
           
           for attempt in range(max_retries + 1):
               try:
                   return func(*args, **kwargs)
               except exceptions as e:
                   last_exception = e
                   if attempt < max_retries:
                       time.sleep(current_delay)
                       current_delay *= backoff
                   else:
                       raise last_exception
           
           return None  # Should never reach here
       return wrapper
   return decorator


def count_words(text: str) -> int:
   """Count words in text."""
   if not text:
       return 0
   
   # Simple word counting - split on whitespace and filter empty strings
   words = [word for word in text.split() if word.strip()]
   return len(words)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
   """Truncate text to specified length with optional suffix."""
   if len(text) <= max_length:
       return text
   
   if len(suffix) >= max_length:
       return text[:max_length]
   
   return text[:max_length - len(suffix)] + suffix


def extract_urls(text: str) -> List[str]:
   """Extract URLs from text."""
   url_pattern = re.compile(
       r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
   )
   return url_pattern.findall(text)


def clean_text(text: str) -> str:
   """Basic text cleaning."""
   if not text:
       return ""
   
   # Remove extra whitespace
   text = re.sub(r'\s+', ' ', text)
   
   # Remove leading/trailing whitespace
   text = text.strip()
   
   return text


def validate_url(url: str) -> bool:
   """Validate if string is a valid URL."""
   url_pattern = re.compile(
       r'^https?://'  # http:// or https://
       r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
       r'localhost|'  # localhost...
       r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
       r'(?::\d+)?'  # optional port
       r'(?:/?|[/?]\S+)$', re.IGNORECASE)
   
   return url_pattern.match(url) is not None


class Timer:
   """Simple timer context manager."""
   
   def __init__(self, name: str = "Operation"):
       """Initialize timer with optional name."""
       self.name = name
       self.start_time = None
       self.end_time = None
   
   def __enter__(self):
       """Start timing."""
       self.start_time = time.time()
       return self
   
   def __exit__(self, exc_type, exc_val, exc_tb):
       """Stop timing."""
       self.end_time = time.time()
   
   @property
   def elapsed(self) -> float:
       """Get elapsed time in seconds."""
       if self.start_time is None:
           return 0.0
       
       end = self.end_time if self.end_time else time.time()
       return end - self.start_time
   
   def __str__(self) -> str:
       """String representation of timer."""
       return f"{self.name}: {format_duration(self.elapsed)}"
