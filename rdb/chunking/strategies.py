"""
Chunking strategies for different chunk sizes.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..chunking.chunker import Chunk
from ..config.settings import Config


class ChunkingStrategy(ABC):
   """Abstract base class for chunking strategies."""
   
   def __init__(self, config: Config):
       """Initialize chunking strategy with configuration."""
       self.config = config
   
   @abstractmethod
   def create_chunks(self, page_title: str, url: str, sections: List[Dict]) -> List[Chunk]:
       """Create chunks from document sections."""
       pass
   
   def _build_section_path(self, sections: List[Dict], current_section: Dict) -> str:
       """Build hierarchical section path."""
       # For now, just use the section title
       # TODO: Implement proper hierarchy building if needed
       return current_section.get('title', 'Untitled')


class SmallChunkStrategy(ChunkingStrategy):
   """Strategy for creating small chunks (paragraphs/logical units)."""
   
   def create_chunks(self, page_title: str, url: str, sections: List[Dict]) -> List[Chunk]:
       """Create small chunks by splitting sections into logical units."""
       chunks = []
       
       for section in sections:
           section_title = section.get('title', 'Untitled')
           section_content = section.get('content', '')
           section_level = section.get('level', 1)
           
           if not section_content.strip():
               continue
           
           section_path = self._build_section_path(sections, section)
           section_anchor = section_title.replace(' ', '_').replace('/', '')
           section_url = url + f"#{section_anchor}"
           
           # Split section into small units
           small_units = self._split_into_small_units(section_content)
           
           for unit in small_units:
               if not unit.strip():
                   continue
               
               # Create chunk text with context
               chunk_text = f"{page_title} - {section_path}: {unit}"
               
               chunks.append(Chunk(
                   page_title=page_title,
                   section_path=section_path,
                   content=unit,
                   chunk_text=chunk_text,
                   url=section_url,
                   chunk_type="small",
                   section_level=section_level
               ))
       
       return chunks
   
   def _split_into_small_units(self, content: str) -> List[str]:
       """Split section content into small logical units."""
       # Split by double newlines (paragraphs)
       paragraphs = content.split('\n\n')
       
       units = []
       current_unit = ""
       
       for paragraph in paragraphs:
           paragraph = paragraph.strip()
           if not paragraph:
               continue
           
           # Handle code blocks specially
           if '```' in paragraph or paragraph.startswith('$ ') or paragraph.startswith('# '):
               # Code block - keep with surrounding context
               if current_unit:
                   units.append(current_unit + '\n\n' + paragraph)
                   current_unit = ""
               else:
                   # Check if previous unit is short, merge with it
                   if units and len(units[-1]) < 200:
                       units[-1] = units[-1] + '\n\n' + paragraph
                   else:
                       units.append(paragraph)
           else:
               # Regular paragraph
               if current_unit and len(current_unit + paragraph) > self.config.chunk_size_small:
                   # Current unit would be too long, save and start new
                   units.append(current_unit)
                   current_unit = paragraph
               else:
                   # Add to current unit
                   if current_unit:
                       current_unit += '\n\n' + paragraph
                   else:
                       current_unit = paragraph
       
       # Save final unit
       if current_unit:
           units.append(current_unit)
       
       return units


class MediumChunkStrategy(ChunkingStrategy):
   """Strategy for creating medium chunks (full sections)."""
   
   def create_chunks(self, page_title: str, url: str, sections: List[Dict]) -> List[Chunk]:
       """Create medium chunks from individual sections."""
       chunks = []
       
       for section in sections:
           section_title = section.get('title', 'Untitled')
           section_content = section.get('content', '')
           section_level = section.get('level', 1)
           
           if not section_content.strip():
               continue
           
           # Build section path
           section_path = self._build_section_path(sections, section)
           
           # Create chunk text with full context
           chunk_text = f"{page_title} - {section_path}: {section_content}"
           
           # Create URL with section anchor
           section_anchor = section_title.replace(' ', '_').replace('/', '')
           section_url = url + f"#{section_anchor}"
           
           chunks.append(Chunk(
               page_title=page_title,
               section_path=section_path,
               content=section_content,
               chunk_text=chunk_text,
               url=section_url,
               chunk_type="medium",
               section_level=section_level
           ))
       
       return chunks


class LargeChunkStrategy(ChunkingStrategy):
   """Strategy for creating large chunks (grouped sections or full pages)."""
   
   def create_chunks(self, page_title: str, url: str, sections: List[Dict]) -> List[Chunk]:
       """Create large chunks by grouping sections or using entire page."""
       chunks = []
       
       # Get major sections (level 1 and 2)
       major_sections = [s for s in sections if s.get('level', 1) <= 2]
       
       if len(major_sections) <= 3:
           # Small page - use entire page as one large chunk
           all_content = "\n\n".join([s.get('content', '') for s in sections])
           all_titles = " > ".join([s.get('title', '') for s in major_sections[:3]])
           
           chunk_text = f"{page_title}: {all_content}"
           
           chunks.append(Chunk(
               page_title=page_title,
               section_path=all_titles,
               content=all_content,
               chunk_text=chunk_text,
               url=url,
               chunk_type="large",
               section_level=1
           ))
       else:
           # Large page - group related level 1 sections
           current_group = []
           current_group_title = ""
           
           for section in sections:
               if section.get('level', 1) == 1:
                   # Save previous group if exists
                   if current_group:
                       self._save_large_chunk_group(
                           page_title, url, current_group_title, current_group, chunks
                       )
                   # Start new group
                   current_group = [section]
                   current_group_title = section.get('title', '')
               elif section.get('level', 1) == 2 and current_group:
                   # Add to current group
                   current_group.append(section)
           
           # Save final group
           if current_group:
               self._save_large_chunk_group(
                   page_title, url, current_group_title, current_group, chunks
               )
       
       return chunks
   
   def _save_large_chunk_group(self, page_title: str, url: str, group_title: str, 
                              group_sections: List[Dict], chunks: List[Chunk]) -> None:
       """Save a group of sections as a large chunk."""
       group_content = "\n\n".join([s.get('content', '') for s in group_sections])
       
       # Don't create empty chunks
       if not group_content.strip():
           return
       
       chunk_text = f"{page_title} - {group_title}: {group_content}"
       
       chunks.append(Chunk(
           page_title=page_title,
           section_path=group_title,
           content=group_content,
           chunk_text=chunk_text,
           url=url + f"#{group_title.replace(' ', '_')}",
           chunk_type="large",
           section_level=1
       ))
