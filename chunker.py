#!/usr/bin/env python3
"""
Arch Wiki 3-Level Chunker
Processes JSON files and creates small/medium/large chunks with context.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a chunk of documentation at any size level."""
    page_title: str
    section_path: str  # e.g., "Device driver > Check status"
    content: str
    chunk_text: str  # Full text with context for embedding
    url: str
    chunk_type: str  # "small", "medium", "large"
    section_level: int
    

class ArchWikiChunker:
    """Creates 3-level chunks from Arch Wiki JSON files."""
    
    def __init__(self):
        self.chunks: List[Chunk] = []
    
    def process_directory(self, json_dir: str) -> List[Chunk]:
        """Process all JSON files in a directory."""
        json_path = Path(json_dir)
        if not json_path.exists():
            raise FileNotFoundError(f"Directory not found: {json_dir}")
            
        json_files = list(json_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in: {json_dir}")
            
        print(f"Processing {len(json_files)} JSON files...")
        
        for json_file in json_files:
            print(f"  Processing: {json_file.name}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                self._process_document(doc)
            except Exception as e:
                print(f"  Error processing {json_file}: {e}")
                continue
                
        print(f"Created {len(self.chunks)} total chunks")
        return self.chunks
    
    def _process_document(self, doc: Dict[str, Any]) -> None:
        """Process a single JSON document and create all chunk levels."""
        page_title = doc.get('title', 'Unknown')
        url = doc.get('url', '')
        sections = doc.get('sections', [])
        
        # Create large chunks (page-level or major section groups)
        self._create_large_chunks(page_title, url, sections)
        
        # Create medium chunks (individual sections)
        self._create_medium_chunks(page_title, url, sections)
        
        # Create small chunks (paragraphs/logical units within sections)
        self._create_small_chunks(page_title, url, sections)
    
    def _create_large_chunks(self, page_title: str, url: str, sections: List[Dict]) -> None:
        """Create large chunks by grouping sections or using entire page."""
        # Strategy 1: Group by major sections (level 1 and 2)
        major_sections = [s for s in sections if s.get('level', 1) <= 2]
        
        if len(major_sections) <= 3:
            # Small page - use entire page as one large chunk
            all_content = "\n\n".join([s.get('content', '') for s in sections])
            all_titles = " > ".join([s.get('title', '') for s in major_sections[:3]])
            
            chunk_text = f"{page_title}: {all_content}"
            
            self.chunks.append(Chunk(
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
                        self._save_large_chunk_group(page_title, url, current_group_title, current_group)
                    # Start new group
                    current_group = [section]
                    current_group_title = section.get('title', '')
                elif section.get('level', 1) == 2 and current_group:
                    # Add to current group
                    current_group.append(section)
            
            # Save final group
            if current_group:
                self._save_large_chunk_group(page_title, url, current_group_title, current_group)
    
    def _save_large_chunk_group(self, page_title: str, url: str, group_title: str, group_sections: List[Dict]) -> None:
        """Save a group of sections as a large chunk."""
        group_content = "\n\n".join([s.get('content', '') for s in group_sections])
        chunk_text = f"{page_title} - {group_title}: {group_content}"
        
        self.chunks.append(Chunk(
            page_title=page_title,
            section_path=group_title,
            content=group_content,
            chunk_text=chunk_text,
            url=url + f"#{group_title.replace(' ', '_')}",
            chunk_type="large", 
            section_level=1
        ))
    
    def _create_medium_chunks(self, page_title: str, url: str, sections: List[Dict]) -> None:
        """Create medium chunks from individual sections."""
        for section in sections:
            section_title = section.get('title', 'Untitled')
            section_content = section.get('content', '')
            section_level = section.get('level', 1)
            
            if not section_content.strip():
                continue
                
            # Build section path (for nested sections)
            section_path = self._build_section_path(sections, section)
            
            # Create chunk text with full context
            chunk_text = f"{page_title} - {section_path}: {section_content}"
            
            # Create URL with section anchor
            section_anchor = section_title.replace(' ', '_').replace('/', '')
            section_url = url + f"#{section_anchor}"
            
            self.chunks.append(Chunk(
                page_title=page_title,
                section_path=section_path,
                content=section_content,
                chunk_text=chunk_text,
                url=section_url,
                chunk_type="medium",
                section_level=section_level
            ))
    
    def _create_small_chunks(self, page_title: str, url: str, sections: List[Dict]) -> None:
        """Create small chunks by splitting sections into logical units."""
        for section in sections:
            section_title = section.get('title', 'Untitled')
            section_content = section.get('content', '')
            section_level = section.get('level', 1)
            
            if not section_content.strip():
                continue
                
            section_path = self._build_section_path(sections, section)
            section_anchor = section_title.replace(' ', '_').replace('/', '')
            section_url = url + f"#{section_anchor}"
            
            # Split section into smaller logical units
            small_units = self._split_into_small_units(section_content)
            
            for i, unit in enumerate(small_units):
                if not unit.strip():
                    continue
                    
                # Create chunk text with context
                chunk_text = f"{page_title} - {section_path}: {unit}"
                
                self.chunks.append(Chunk(
                    page_title=page_title,
                    section_path=section_path,
                    content=unit,
                    chunk_text=chunk_text,
                    url=section_url,
                    chunk_type="small",
                    section_level=section_level
                ))
    
    def _build_section_path(self, all_sections: List[Dict], current_section: Dict) -> str:
        """Build hierarchical path like 'Device driver > Check status'."""
        # For now, just use the section title
        # TODO: Build actual hierarchy if needed
        return current_section.get('title', 'Untitled')
    
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
                
            # If paragraph has code blocks, keep them with surrounding text
            if '```' in paragraph or paragraph.startswith('$ ') or paragraph.startswith('# '):
                # If we have accumulated text, save it with this code block
                if current_unit:
                    units.append(current_unit + '\n\n' + paragraph)
                    current_unit = ""
                else:
                    # Code block standalone, but look for previous context
                    if units and len(units[-1]) < 200:  # Short previous unit
                        # Merge with previous unit
                        units[-1] = units[-1] + '\n\n' + paragraph
                    else:
                        units.append(paragraph)
            else:
                # Regular text paragraph
                if current_unit and len(current_unit + paragraph) > 300:
                    # Current unit is getting long, save it and start new
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
    
    def save_chunks(self, output_file: str) -> None:
        """Save chunks to JSON file for inspection."""
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append({
                'page_title': chunk.page_title,
                'section_path': chunk.section_path,
                'content': chunk.content,
                'chunk_text': chunk.chunk_text,
                'url': chunk.url,
                'chunk_type': chunk.chunk_type,
                'section_level': chunk.section_level
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks_data)} chunks to {output_file}")
    
    def print_stats(self) -> None:
        """Print chunking statistics."""
        small_count = len([c for c in self.chunks if c.chunk_type == "small"])
        medium_count = len([c for c in self.chunks if c.chunk_type == "medium"])
        large_count = len([c for c in self.chunks if c.chunk_type == "large"])
        
        print("\nChunking Statistics:")
        print(f"  Small chunks:  {small_count}")
        print(f"  Medium chunks: {medium_count}")
        print(f"  Large chunks:  {large_count}")
        print(f"  Total chunks:  {len(self.chunks)}")


def main():
    """Main function to run chunking."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chunk Arch Wiki JSON files')
    parser.add_argument('json_dir', help='Directory containing JSON files')
    parser.add_argument('--output', '-o', default='chunks.json', help='Output file for chunks')
    
    args = parser.parse_args()
    
    chunker = ArchWikiChunker()
    chunks = chunker.process_directory(args.json_dir)
    chunker.print_stats()
    chunker.save_chunks(args.output)
    
    print(f"\nNext step: Run vectorization on {args.output}")


if __name__ == "__main__":
    main()
