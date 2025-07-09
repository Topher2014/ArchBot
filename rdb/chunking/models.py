"""
Data models for chunking module.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a chunk of documentation."""
    page_title: str
    section_path: str
    content: str
    chunk_text: str
    url: str
    chunk_type: str
    section_level: int
