"""
Tests for the chunking module.
"""

import pytest
import json
from pathlib import Path

from rdb.config.settings import Config
from rdb.chunking.chunker import DocumentChunker, Chunk
from rdb.chunking.strategies import SmallChunkStrategy, MediumChunkStrategy, LargeChunkStrategy


class TestChunk:
   """Test cases for Chunk dataclass."""
   
   def test_chunk_creation(self):
       """Test creating a chunk object."""
       chunk = Chunk(
           page_title="Test Page",
           section_path="Section 1",
           content="Test content",
           chunk_text="Test Page - Section 1: Test content",
           url="http://example.com/test#section1",
           chunk_type="medium",
           section_level=2
       )
       
       assert chunk.page_title == "Test Page"
       assert chunk.section_path == "Section 1"
       assert chunk.content == "Test content"
       assert chunk.chunk_type == "medium"
       assert chunk.section_level == 2


class TestSmallChunkStrategy:
   """Test cases for SmallChunkStrategy."""
   
   def setup_method(self):
       """Setup test fixtures."""
       self.config = Config()
       self.config.chunk_size_small = 100  # Small size for testing
       self.strategy = SmallChunkStrategy(self.config)
   
   def test_split_into_small_units(self):
       """Test splitting content into small units."""
       content = """This is the first paragraph.

This is the second paragraph with more content that might be longer.
This is a code block
sudo pacman -S package

This is after the code block."""
        
        units = self.strategy._split_into_small_units(content)
        
        assert len(units) >= 2
        assert any("first paragraph" in unit for unit in units)
        assert any("code block" in unit for unit in units)
    
    def test_create_chunks(self):
        """Test creating small chunks from sections."""
        sections = [
            {
                'title': 'Installation',
                'content': 'Install using pacman.\n\nRun the command below.\n\n```\nsudo pacman -S package\n```',
                'level': 2
            }
        ]
        
        chunks = self.strategy.create_chunks("Test Page", "http://example.com/test", sections)
        
        assert len(chunks) > 0
        assert all(chunk.chunk_type == "small" for chunk in chunks)
        assert all(chunk.page_title == "Test Page" for chunk in chunks)


class TestMediumChunkStrategy:
    """Test cases for MediumChunkStrategy."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config()
        self.strategy = MediumChunkStrategy(self.config)
    
    def test_create_chunks(self):
        """Test creating medium chunks from sections."""
        sections = [
            {
                'title': 'Introduction',
                'content': 'This is the introduction section.',
                'level': 1
            },
            {
                'title': 'Installation',
                'content': 'This is the installation section.',
                'level': 2
            }
        ]
        
        chunks = self.strategy.create_chunks("Test Page", "http://example.com/test", sections)
        
        assert len(chunks) == 2
        assert all(chunk.chunk_type == "medium" for chunk in chunks)
        assert chunks[0].section_path == "Introduction"
        assert chunks[1].section_path == "Installation"


class TestLargeChunkStrategy:
    """Test cases for LargeChunkStrategy."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config()
        self.strategy = LargeChunkStrategy(self.config)
    
    def test_create_chunks_small_page(self):
        """Test creating large chunks for a small page."""
        sections = [
            {'title': 'Introduction', 'content': 'Intro content', 'level': 1},
            {'title': 'Usage', 'content': 'Usage content', 'level': 2}
        ]
        
        chunks = self.strategy.create_chunks("Small Page", "http://example.com/small", sections)
        
        # Should create one large chunk for small page
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "large"
        assert "Intro content" in chunks[0].content
        assert "Usage content" in chunks[0].content
    
    def test_create_chunks_large_page(self):
        """Test creating large chunks for a large page."""
        sections = [
            {'title': 'Introduction', 'content': 'Intro content', 'level': 1},
            {'title': 'Installation', 'content': 'Install content', 'level': 1},
            {'title': 'Prerequisites', 'content': 'Prereq content', 'level': 2},
            {'title': 'Configuration', 'content': 'Config content', 'level': 1},
            {'title': 'Advanced', 'content': 'Advanced content', 'level': 2}
        ]
        
        chunks = self.strategy.create_chunks("Large Page", "http://example.com/large", sections)
        
        # Should create multiple large chunks by grouping level 1 sections
        assert len(chunks) > 1
        assert all(chunk.chunk_type == "large" for chunk in chunks)


class TestDocumentChunker:
    """Test cases for DocumentChunker."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = Config()
        self.chunker = DocumentChunker(self.config)
    
    def test_process_document(self):
        """Test processing a single document."""
        doc = {
            'title': 'Test Document',
            'url': 'http://example.com/test',
            'sections': [
                {
                    'title': 'Introduction',
                    'content': 'This is the introduction with enough content to create multiple chunks.',
                    'level': 1
                },
                {
                    'title': 'Details',
                    'content': 'This section has more detailed information about the topic.',
                    'level': 2
                }
            ]
        }
        
        initial_count = len(self.chunker.chunks)
        self.chunker._process_document(doc)
        
        # Should have created chunks
        assert len(self.chunker.chunks) > initial_count
        
        # Should have different chunk types
        chunk_types = set(chunk.chunk_type for chunk in self.chunker.chunks)
        assert len(chunk_types) > 1
    
    def test_process_directory(self, tmp_path):
        """Test processing a directory of JSON files."""
        # Create test JSON files
        doc1 = {
            'title': 'Document 1',
            'url': 'http://example.com/doc1',
            'sections': [
                {'title': 'Section 1', 'content': 'Content 1', 'level': 1}
            ]
        }
        
        doc2 = {
            'title': 'Document 2', 
            'url': 'http://example.com/doc2',
            'sections': [
                {'title': 'Section 2', 'content': 'Content 2', 'level': 1}
            ]
        }
        
        # Save test files
        with open(tmp_path / "doc1.json", 'w') as f:
            json.dump(doc1, f)
        
        with open(tmp_path / "doc2.json", 'w') as f:
            json.dump(doc2, f)
        
        # Process directory
        chunks = self.chunker.process_directory(str(tmp_path))
        
        assert len(chunks) > 0
        
        # Should have chunks from both documents
        page_titles = set(chunk.page_title for chunk in chunks)
        assert "Document 1" in page_titles
        assert "Document 2" in page_titles
    
    def test_save_and_load_chunks(self, tmp_path):
        """Test saving and loading chunks."""
        # Create some test chunks
        self.chunker.chunks = [
            Chunk(
                page_title="Test Page",
                section_path="Section 1",
                content="Test content",
                chunk_text="Test Page - Section 1: Test content",
                url="http://example.com/test#section1",
                chunk_type="medium",
                section_level=2
            )
        ]
        
        # Save chunks
        output_file = tmp_path / "test_chunks.json"
        self.chunker.save_chunks(str(output_file))
        
        assert output_file.exists()
        
        # Load chunks
        new_chunker = DocumentChunker(self.config)
        loaded_chunks = new_chunker.load_chunks(str(output_file))
        
        assert len(loaded_chunks) == 1
        assert loaded_chunks[0].page_title == "Test Page"
        assert loaded_chunks[0].chunk_type == "medium"
    
    def test_get_stats(self):
        """Test getting chunking statistics."""
        # Add some test chunks
        self.chunker.chunks = [
            Chunk("Page1", "Sec1", "Content1", "Text1", "URL1", "small", 1),
            Chunk("Page1", "Sec2", "Content2", "Text2", "URL2", "medium", 2),
            Chunk("Page2", "Sec1", "Content3", "Text3", "URL3", "large", 1),
            Chunk("Page2", "Sec2", "Content4", "Text4", "URL4", "small", 2)
        ]
        
        stats = self.chunker.get_stats()
        
        assert stats['small'] == 2
        assert stats['medium'] == 1
        assert stats['large'] == 1
        assert stats['total'] == 4


class TestChunkingIntegration:
    """Integration tests for the chunking module."""
    
    def test_full_chunking_workflow(self, tmp_path):
        """Test complete chunking workflow."""
        config = Config(data_dir=str(tmp_path))
        chunker = DocumentChunker(config)
        
        # Create test document with various content types
        doc = {
            'title': 'Complete Test Document',
            'url': 'http://example.com/complete-test',
            'sections': [
                {
                    'title': 'Introduction',
                    'content': 'This is a comprehensive introduction to the topic. ' * 10,
                    'level': 1
                },
                {
                    'title': 'Installation',
                    'content': 'Install the software using these steps.\n\n```\nsudo pacman -S software\n```\n\nThen configure it.',
                    'level': 2
                },
                {
                    'title': 'Configuration',
                    'content': 'Configure the software by editing files.\n\nEdit /etc/config.conf:\n\n- Option 1\n- Option 2',
                    'level': 2
                },
                {
                    'title': 'Troubleshooting',
                    'content': 'Common problems and solutions. ' * 20,
                    'level': 1
                }
            ]
        }
        
        # Save test document
        doc_file = tmp_path / "complete_test.json"
        with open(doc_file, 'w') as f:
            json.dump(doc, f)
        
        # Process the document
        chunks = chunker.process_directory(str(tmp_path))
        
        # Verify results
        assert len(chunks) > 0
        
        # Should have all three chunk types
        chunk_types = set(chunk.chunk_type for chunk in chunks)
        assert "small" in chunk_types
        assert "medium" in chunk_types
        assert "large" in chunk_types
        
        # All chunks should be from the same document
        assert all(chunk.page_title == "Complete Test Document" for chunk in chunks)
        
        # Should have preserved code blocks and lists
        content_text = " ".join(chunk.content for chunk in chunks)
        assert "pacman -S" in content_text
        assert "Option 1" in content_text
        
        # Test saving and loading
        chunker.save_chunks()
        
        new_chunker = DocumentChunker(config)
        loaded_chunks = new_chunker.load_chunks()
        
        assert len(loaded_chunks) == len(chunks)
