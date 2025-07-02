"""
Content parser for extracting structured data from HTML.
"""

import re
from bs4 import BeautifulSoup, Tag
from urllib.parse import unquote
from typing import Dict, List, Any, Optional
import logging

from ..utils.logging import get_logger


class ContentParser:
   """Parser for extracting structured content from wiki pages."""
   
   def __init__(self):
       """Initialize content parser."""
       self.logger = get_logger(__name__)
   
   def clean_text(self, text: str) -> str:
       """Clean up text by removing unwanted elements."""
       if not text:
           return ""
       
       # Remove [edit] links
       text = re.sub(r'\[edit\]', '', text)
       # Remove multiple spaces
       text = re.sub(r'\s+', ' ', text)
       return text.strip()
   
   def extract_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict[str, Any]]:
       """Extract structured content from a wiki page."""
       try:
           page_title = unquote(url.split('/title/')[-1].replace('_', ' '))
           
           # Get main content
           content_div = soup.find('div', {'id': 'mw-content-text'})
           if not content_div:
               self.logger.warning(f"No content div found for {page_title}")
               return None
           
           sections = self._extract_sections(content_div)
           
           if not sections:
               self.logger.warning(f"No sections extracted for {page_title}")
               return None
           
           return {
               "title": page_title,
               "url": url,
               "sections": sections
           }
           
       except Exception as e:
           self.logger.error(f"Error extracting content from {url}: {e}")
           return None
   
   def _extract_sections(self, content_div: Tag) -> List[Dict[str, Any]]:
       """Extract sections from content div."""
       sections = []
       current_section = {"title": "Introduction", "level": 1, "content": ""}
       
       # Find all content elements
       elements = content_div.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre', 'ul', 'ol', 'div', 'table'], recursive=True)
       
       for element in elements:
           if self._is_heading(element):
               # Save previous section if it has content
               if current_section["content"].strip():
                   sections.append(current_section.copy())
               
               # Start new section
               level = int(element.name[1])
               title = self.clean_text(element.get_text())
               
               current_section = {
                   "title": title,
                   "level": level,
                   "content": ""
               }
           else:
               # Process content element
               content = self._process_element(element)
               if content:
                   current_section["content"] += content + "\n\n"
       
       # Add final section
       if current_section["content"].strip():
           sections.append(current_section)
       
       return sections
   
   def _is_heading(self, element: Tag) -> bool:
       """Check if element is a heading."""
       return element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
   
   def _process_element(self, element: Tag) -> str:
       """Process individual content element."""
       if element.name == 'p':
           return self._process_paragraph(element)
       elif element.name == 'pre':
           return self._process_code_block(element)
       elif element.name in ['ul', 'ol']:
           return self._process_list(element)
       elif element.name == 'div':
           return self._process_div(element)
       elif element.name == 'table':
           return self._process_table(element)
       else:
           return ""
   
   def _process_paragraph(self, element: Tag) -> str:
       """Process paragraph element."""
       text = self.clean_text(element.get_text())
       return text if text else ""
   
   def _process_code_block(self, element: Tag) -> str:
       """Process code block element."""
       code = element.get_text().strip()
       return f"```\n{code}\n```" if code else ""
   
   def _process_list(self, element: Tag) -> str:
       """Process list element."""
       items = []
       for li in element.find_all('li', recursive=False):
           text = self.clean_text(li.get_text())
           if text:
               prefix = "- " if element.name == 'ul' else "1. "
               items.append(f"{prefix}{text}")
       
       return "\n".join(items) if items else ""
   
   def _process_div(self, element: Tag) -> str:
       """Process div element (focus on info boxes)."""
       classes = element.get('class', [])
       
       # Handle info/warning/note boxes
       if any(cls in classes for cls in ['archwiki-template-box', 'archwiki-template-message']):
           box_title = element.find('b')
           box_text = ""
           
           if box_title:
               box_text = f"**{box_title.get_text()}** "
           
           box_content = element.find('p')
           if box_content:
               box_text += self.clean_text(box_content.get_text())
           
           return f"Note: {box_text}" if box_text.strip() else ""
       
       return ""
   
   def _process_table(self, element: Tag) -> str:
       """Process table element."""
       rows = []
       for row in element.find_all('tr'):
           cells = row.find_all(['th', 'td'])
           if cells:
               row_content = " | ".join([self.clean_text(cell.get_text()) for cell in cells])
               if row_content.strip():
                   rows.append(row_content)
       
       if len(rows) > 1:  # Only include non-empty tables
           return "Table content:\n" + "\n".join(rows)
       
       return ""
