"""
Database manager for persistent storage.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..config.settings import Config
from ..utils.logging import get_logger


class DatabaseManager:
   """Manages persistent storage for RDB data."""
   
   def __init__(self, config: Config):
       """Initialize database manager with configuration."""
       self.config = config
       self.logger = get_logger(__name__)
       self.db_path = config.data_dir / "rdb.db"
       self._init_database()
   
   def _init_database(self):
       """Initialize database tables if they don't exist."""
       with sqlite3.connect(self.db_path) as conn:
           cursor = conn.cursor()
           
           # Scraping sessions table
           cursor.execute("""
               CREATE TABLE IF NOT EXISTS scraping_sessions (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   started_at TIMESTAMP,
                   completed_at TIMESTAMP,
                   total_pages INTEGER,
                   success_count INTEGER,
                   error_count INTEGER,
                   skip_count INTEGER,
                   status TEXT,
                   config_snapshot TEXT
               )
           """)
           
           # Indexing sessions table
           cursor.execute("""
               CREATE TABLE IF NOT EXISTS indexing_sessions (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   started_at TIMESTAMP,
                   completed_at TIMESTAMP,
                   input_dir TEXT,
                   output_dir TEXT,
                   total_chunks INTEGER,
                   embedding_model TEXT,
                   status TEXT,
                   config_snapshot TEXT
               )
           """)
           
           # Search history table
           cursor.execute("""
               CREATE TABLE IF NOT EXISTS search_history (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   timestamp TIMESTAMP,
                   original_query TEXT,
                   refined_query TEXT,
                   top_k INTEGER,
                   results_count INTEGER,
                   search_time_ms INTEGER,
                   query_refinement_enabled BOOLEAN
               )
           """)
           
           # Page metadata table
           cursor.execute("""
               CREATE TABLE IF NOT EXISTS page_metadata (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   page_title TEXT UNIQUE,
                   url TEXT,
                   last_scraped TIMESTAMP,
                   content_hash TEXT,
                   section_count INTEGER,
                   word_count INTEGER
               )
           """)
           
           conn.commit()
           self.logger.info(f"Database initialized at {self.db_path}")
   
   def log_scraping_session(self, session_data: Dict[str, Any]) -> int:
       """Log a scraping session to the database."""
       with sqlite3.connect(self.db_path) as conn:
           cursor = conn.cursor()
           
           cursor.execute("""
               INSERT INTO scraping_sessions 
               (started_at, completed_at, total_pages, success_count, error_count, 
                skip_count, status, config_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           """, (
               session_data.get('started_at'),
               session_data.get('completed_at'),
               session_data.get('total_pages', 0),
               session_data.get('success_count', 0),
               session_data.get('error_count', 0),
               session_data.get('skip_count', 0),
               session_data.get('status', 'unknown'),
               json.dumps(session_data.get('config', {}))
           ))
           
           session_id = cursor.lastrowid
           conn.commit()
           
           self.logger.info(f"Logged scraping session {session_id}")
           return session_id
   
   def log_indexing_session(self, session_data: Dict[str, Any]) -> int:
       """Log an indexing session to the database."""
       with sqlite3.connect(self.db_path) as conn:
           cursor = conn.cursor()
           
           cursor.execute("""
               INSERT INTO indexing_sessions 
               (started_at, completed_at, input_dir, output_dir, total_chunks, 
                embedding_model, status, config_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           """, (
               session_data.get('started_at'),
               session_data.get('completed_at'),
               session_data.get('input_dir'),
               session_data.get('output_dir'),
               session_data.get('total_chunks', 0),
               session_data.get('embedding_model'),
               session_data.get('status', 'unknown'),
               json.dumps(session_data.get('config', {}))
           ))
           
           session_id = cursor.lastrowid
           conn.commit()
           
           self.logger.info(f"Logged indexing session {session_id}")
           return session_id
   
   def log_search(self, search_data: Dict[str, Any]) -> int:
       """Log a search query to the database."""
       with sqlite3.connect(self.db_path) as conn:
           cursor = conn.cursor()
           
           cursor.execute("""
               INSERT INTO search_history 
               (timestamp, original_query, refined_query, top_k, results_count, 
                search_time_ms, query_refinement_enabled)
               VALUES (?, ?, ?, ?, ?, ?, ?)
           """, (
               datetime.now(),
               search_data.get('original_query'),
               search_data.get('refined_query'),
               search_data.get('top_k', 5),
               search_data.get('results_count', 0),
               search_data.get('search_time_ms', 0),
               search_data.get('query_refinement_enabled', False)
           ))
           
           search_id = cursor.lastrowid
           conn.commit()
           
           return search_id
   
   def update_page_metadata(self, page_data: Dict[str, Any]):
       """Update or insert page metadata."""
       with sqlite3.connect(self.db_path) as conn:
           cursor = conn.cursor()
           
           cursor.execute("""
               INSERT OR REPLACE INTO page_metadata 
               (page_title, url, last_scraped, content_hash, section_count, word_count)
               VALUES (?, ?, ?, ?, ?, ?)
           """, (
               page_data.get('page_title'),
               page_data.get('url'),
               datetime.now(),
               page_data.get('content_hash'),
               page_data.get('section_count', 0),
               page_data.get('word_count', 0)
           ))
           
           conn.commit()
   
   def get_recent_searches(self, limit: int = 50) -> List[Dict[str, Any]]:
       """Get recent search history."""
       with sqlite3.connect(self.db_path) as conn:
           cursor = conn.cursor()
           
           cursor.execute("""
               SELECT * FROM search_history 
               ORDER BY timestamp DESC 
               LIMIT ?
           """, (limit,))
           
           columns = [desc[0] for desc in cursor.description]
           results = []
           
           for row in cursor.fetchall():
               results.append(dict(zip(columns, row)))
           
           return results
   
   def get_scraping_stats(self) -> Dict[str, Any]:
       """Get scraping statistics."""
       with sqlite3.connect(self.db_path) as conn:
           cursor = conn.cursor()
           
           cursor.execute("""
               SELECT 
                   COUNT(*) as total_sessions,
                   SUM(success_count) as total_pages_scraped,
                   SUM(error_count) as total_errors,
                   MAX(completed_at) as last_scrape
               FROM scraping_sessions 
               WHERE status = 'completed'
           """)
           
           row = cursor.fetchone()
           
           return {
               'total_sessions': row[0] or 0,
               'total_pages_scraped': row[1] or 0,
               'total_errors': row[2] or 0,
               'last_scrape': row[3]
           }
   
   def get_search_stats(self) -> Dict[str, Any]:
       """Get search statistics."""
       with sqlite3.connect(self.db_path) as conn:
           cursor = conn.cursor()
           
           cursor.execute("""
               SELECT 
                   COUNT(*) as total_searches,
                   AVG(search_time_ms) as avg_search_time,
                   COUNT(CASE WHEN query_refinement_enabled THEN 1 END) as refined_searches,
                   MAX(timestamp) as last_search
               FROM search_history
           """)
           
           row = cursor.fetchone()
           
           return {
               'total_searches': row[0] or 0,
               'avg_search_time_ms': row[1] or 0,
               'refined_searches': row[2] or 0,
               'last_search': row[3]
           }
   
   def cleanup_old_data(self, days_old: int = 30):
       """Remove old search history and session data."""
       with sqlite3.connect(self.db_path) as conn:
           cursor = conn.cursor()
           
           cutoff_date = datetime.now().replace(day=datetime.now().day - days_old)
           
           cursor.execute("""
               DELETE FROM search_history 
               WHERE timestamp < ?
           """, (cutoff_date,))
           
           deleted_searches = cursor.rowcount
           
           cursor.execute("""
               DELETE FROM scraping_sessions 
               WHERE completed_at < ?
           """, (cutoff_date,))
           
           deleted_sessions = cursor.rowcount
           
           conn.commit()
           
           self.logger.info(f"Cleaned up {deleted_searches} old searches and {deleted_sessions} old sessions")
