"""
Logging utilities for RDB.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, 
                enable_console: bool = True) -> None:
   """Setup logging configuration for RDB."""
   
   # Convert string level to logging constant
   numeric_level = getattr(logging, log_level.upper(), logging.INFO)
   
   # Create formatter
   formatter = logging.Formatter(
       '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       datefmt='%Y-%m-%d %H:%M:%S'
   )
   
   # Get root logger
   root_logger = logging.getLogger()
   root_logger.setLevel(numeric_level)
   
   # Clear existing handlers
   root_logger.handlers = []
   
   # Console handler
   if enable_console:
       console_handler = logging.StreamHandler(sys.stdout)
       console_handler.setLevel(numeric_level)
       console_handler.setFormatter(formatter)
       root_logger.addHandler(console_handler)
   
   # File handler
   if log_file:
       log_path = Path(log_file)
       log_path.parent.mkdir(parents=True, exist_ok=True)
       
       file_handler = logging.FileHandler(log_path, encoding='utf-8')
       file_handler.setLevel(numeric_level)
       file_handler.setFormatter(formatter)
       root_logger.addHandler(file_handler)
   
   # Reduce noise from external libraries
   logging.getLogger('urllib3').setLevel(logging.WARNING)
   logging.getLogger('requests').setLevel(logging.WARNING)
   logging.getLogger('transformers').setLevel(logging.WARNING)
   logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
   """Get a logger instance for the given name."""
   return logging.getLogger(name)


class LogCapture:
   """Context manager to capture log output for testing."""
   
   def __init__(self, logger_name: str = None, level: int = logging.INFO):
       """Initialize log capture."""
       self.logger_name = logger_name or 'rdb'
       self.level = level
       self.handler = None
       self.logs = []
   
   def __enter__(self):
       """Start capturing logs."""
       self.handler = logging.Handler()
       self.handler.setLevel(self.level)
       self.handler.emit = lambda record: self.logs.append(record)
       
       logger = logging.getLogger(self.logger_name)
       logger.addHandler(self.handler)
       
       return self
   
   def __exit__(self, exc_type, exc_val, exc_tb):
       """Stop capturing logs."""
       if self.handler:
           logger = logging.getLogger(self.logger_name)
           logger.removeHandler(self.handler)
   
   def get_messages(self, level: Optional[int] = None) -> list:
       """Get captured log messages, optionally filtered by level."""
       if level is None:
           return [record.getMessage() for record in self.logs]
       else:
           return [record.getMessage() for record in self.logs if record.levelno >= level]


def log_function_call(func):
   """Decorator to log function entry and exit."""
   def wrapper(*args, **kwargs):
       logger = get_logger(func.__module__)
       logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
       
       try:
           result = func(*args, **kwargs)
           logger.debug(f"Exiting {func.__name__} successfully")
           return result
       except Exception as e:
           logger.error(f"Error in {func.__name__}: {e}")
           raise
   
   return wrapper


def log_performance(func):
   """Decorator to log function performance."""
   import time
   
   def wrapper(*args, **kwargs):
       logger = get_logger(func.__module__)
       start_time = time.time()
       
       try:
           result = func(*args, **kwargs)
           duration = time.time() - start_time
           logger.info(f"{func.__name__} completed in {duration:.2f}s")
           return result
       except Exception as e:
           duration = time.time() - start_time
           logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
           raise
   
   return wrapper
