"""
Tests for the embedding module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rdb.config.settings import Config
from rdb.embedding.embedder import DocumentEmbedder
from rdb.embedding.models import EmbeddingModel


class TestEmbeddingModel:
   """Test cases for EmbeddingModel."""
   
   def setup_method(self):
       """Setup test fixtures."""
       self.model_name = 'test-model'
       self.device = 'cpu'
   
   @patch('rdb.embedding.models.SentenceTransformer')
   def test_init(self, mock_sentence_transformer):
       """Test EmbeddingModel initialization."""
       # Mock the SentenceTransformer
       mock_model = Mock()
       mock_model.get_sentence_embedding_dimension.return_value = 768
       mock_model.max_seq_length = 512
       mock_model.device = 'cpu'
       mock_sentence_transformer.return_value = mock_model
       
       embedding_model = EmbeddingModel(self.model_name, device=self.device)
       
       assert embedding_model.model_name == self.model_name
       assert embedding_model.device == self.device
       assert embedding_model.dimension == 768
       assert embedding_model.max_seq_length == 512
   
   @patch('rdb.embedding.models.SentenceTransformer')
   def test_encode(self, mock_sentence_transformer):
       """Test text encoding."""
       # Mock the SentenceTransformer
       mock_model = Mock()
       mock_model.get_sentence_embedding_dimension.return_value = 384
       mock_model.max_seq_length = 512
       mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
       mock_sentence_transformer.return_value = mock_model
       
       embedding_model = EmbeddingModel(self.model_name, device=self.device)
       
       result = embedding_model.encode("test text")
       
       assert isinstance(result, np.ndarray)
       assert result.shape == (1, 3)
       mock_model.encode.assert_called_once()
   
   @patch('rdb.embedding.models.SentenceTransformer')
   def test_encode_query(self, mock_sentence_transformer):
       """Test query encoding with proper prefix."""
       # Mock the SentenceTransformer
       mock_model = Mock()
       mock_model.get_sentence_embedding_dimension.return_value = 384
       mock_model.max_seq_length = 512
       mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
       mock_sentence_transformer.return_value = mock_model
       
       # Test with e5 model (should add prefix)
       embedding_model = EmbeddingModel('intfloat/e5-large-v2', device=self.device)
       
       result = embedding_model.encode_query("test query")
       
       # Should call encode with "query: " prefix
       mock_model.encode.assert_called_with(["query: test query"])
       assert isinstance(result, np.ndarray)
   
   @patch('rdb.embedding.models.SentenceTransformer')
   def test_encode_passage(self, mock_sentence_transformer):
       """Test passage encoding with proper prefix."""
       # Mock the SentenceTransformer
       mock_model = Mock()
       mock_model.get_sentence_embedding_dimension.return_value = 384
       mock_model.max_seq_length = 512
       mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
       mock_sentence_transformer.return_value = mock_model
       
       # Test with e5 model (should add prefix)
       embedding_model = EmbeddingModel('intfloat/e5-large-v2', device=self.device)
       
       result = embedding_model.encode_passage("test passage")
       
       # Should call encode with "passage: " prefix
       mock_model.encode.assert_called_with(["passage: test passage"])
       assert isinstance(result, np.ndarray)
   
   @patch('rdb.embedding.models.SentenceTransformer')
   def test_get_info(self, mock_sentence_transformer):
       """Test getting embedding model information."""
       # Mock the SentenceTransformer
       mock_model = Mock()
       mock_model.get_sentence_embedding_dimension.return_value = 768
       mock_model.max_seq_length = 512
       mock_model.device = 'cpu'
       mock_sentence_transformer.return_value = mock_model
       
       with patch('torch.cuda.is_available', return_value=False):
           embedding_model = EmbeddingModel('test-model', device='cpu')
           info = embedding_model.get_info()
       
       assert info['model_name'] == 'test-model'
       assert info['device'] == 'cpu'
       assert info['dimension'] == 768
       assert info['max_seq_length'] == 512
       assert info['is_cuda_available'] == False


class TestDocumentEmbedder:
   """Test cases for DocumentEmbedder."""
   
   def setup_method(self):
       """Setup test fixtures."""
       self.config = Config()
       
   @patch('rdb.embedding.models.EmbeddingModel')
   def test_init(self, mock_embedding_model):
       """Test DocumentEmbedder initialization."""
       embedder = DocumentEmbedder(self.config)
       
       assert embedder.config == self.config
       mock_embedding_model.assert_called_once()
   
   @patch('rdb.embedding.models.EmbeddingModel')
   def test_load_chunks(self, mock_embedding_model, tmp_path):
       """Test loading chunks from file."""
       # Create test chunks file
       test_chunks = [
           {
               'page_title': 'Test Page',
               'section_path': 'Test Section',
               'content': 'Test content',
               'chunk_text': 'Test Page - Test Section: Test content',
               'url': 'http://test.com',
               'chunk_type': 'small',
               'section_level': 1
           }
       ]
       
       chunks_file = tmp_path / "chunks.json"
       import json
       with open(chunks_file, 'w') as f:
           json.dump(test_chunks, f)
       
       embedder = DocumentEmbedder(self.config)
       loaded_chunks = embedder.load_chunks(str(chunks_file))
       
       assert len(loaded_chunks) == 1
       assert loaded_chunks[0]['page_title'] == 'Test Page'
   
   @patch('rdb.embedding.models.EmbeddingModel')
   @patch('faiss.IndexFlatIP')
   @patch('faiss.normalize_L2')
   def test_create_embeddings(self, mock_normalize, mock_index_class, mock_embedding_model):
       """Test creating embeddings from chunks."""
       # Mock embedding model
       mock_model = Mock()
       mock_model.encode.return_value = np.random.rand(2, 384)
       mock_embedding_model.return_value = mock_model
       
       embedder = DocumentEmbedder(self.config)
       
       # Test chunks
       test_chunks = [
           {
               'page_title': 'Page 1',
               'section_path': 'Section 1',
               'content': 'Content 1',
               'chunk_text': 'Page 1 - Section 1: Content 1',
               'url': 'http://test1.com',
               'chunk_type': 'small',
               'section_level': 1
           },
           {
               'page_title': 'Page 2',
               'section_path': 'Section 2',
               'content': 'Content 2',
               'chunk_text': 'Page 2 - Section 2: Content 2',
               'url': 'http://test2.com',
               'chunk_type': 'medium',
               'section_level': 2
           }
       ]
       
       embeddings = embedder.create_embeddings(test_chunks)
       
       assert isinstance(embeddings, np.ndarray)
       assert embeddings.shape[0] == 2  # 2 chunks
       mock_model.encode.assert_called()
   
   @patch('rdb.embedding.models.EmbeddingModel')
   @patch('faiss.IndexFlatIP')
   @patch('faiss.normalize_L2')
   def test_build_index(self, mock_normalize, mock_index_class, mock_embedding_model):
       """Test building FAISS index."""
       # Mock FAISS index
       mock_index = Mock()
       mock_index.ntotal = 2
       mock_index_class.return_value = mock_index
       
       embedder = DocumentEmbedder(self.config)
       
       # Test embeddings
       test_embeddings = np.random.rand(2, 384)
       
       index = embedder.build_index(test_embeddings)
       
       assert index == mock_index
       mock_index.add.assert_called_once()
       mock_normalize.assert_called_once()
   
   @patch('rdb.embedding.models.EmbeddingModel')
   @patch('faiss.IndexFlatIP')
   @patch('faiss.normalize_L2')
   def test_end_to_end_embedding_process(self, mock_normalize, mock_index_class, mock_embedding_model):
       """Test end-to-end embedding process with realistic data."""
       # Mock embedding model
       mock_model = Mock()
       mock_model.encode.return_value = np.random.rand(3, 384)  # 3 chunks, 384 dimensions
       mock_embedding_model.return_value = mock_model
       
       # Mock FAISS index
       mock_index = Mock()
       mock_index.ntotal = 3
       mock_index_class.return_value = mock_index
       
       embedder = DocumentEmbedder(self.config)
       
       # Test chunks with realistic content
       test_chunks = [
           {
               'page_title': 'Arch Linux Installation',
               'section_path': 'Pre-installation',
               'content': 'Download the Arch Linux ISO and create a bootable USB drive.',
               'chunk_text': 'passage: Arch Linux Installation - Pre-installation: Download the Arch Linux ISO and create a bootable USB drive.',
               'url': 'https://wiki.archlinux.org/title/Installation_guide#Pre-installation',
               'chunk_type': 'medium',
               'section_level': 2
           },
           {
               'page_title': 'Arch Linux Installation', 
               'section_path': 'Boot the live environment',
               'content': 'Boot from the USB drive and verify the boot mode.',
               'chunk_text': 'passage: Arch Linux Installation - Boot the live environment: Boot from the USB drive and verify the boot mode.',
               'url': 'https://wiki.archlinux.org/title/Installation_guide#Boot_the_live_environment',
               'chunk_type': 'small',
               'section_level': 3
           },
           {
               'page_title': 'Network Configuration',
               'section_path': 'Wireless',
               'content': 'Configure wireless network using iwctl or NetworkManager.',
               'chunk_text': 'passage: Network Configuration - Wireless: Configure wireless network using iwctl or NetworkManager.',
               'url': 'https://wiki.archlinux.org/title/Network_configuration#Wireless',
               'chunk_type': 'medium',
               'section_level': 2
           }
       ]
       
       # Process embeddings
       embeddings = embedder.create_embeddings(test_chunks)
       
       # Verify embeddings were created
       assert embeddings.shape == (3, 384)
       assert isinstance(embeddings, np.ndarray)
       
       # Build index
       index = embedder.build_index(embeddings)
       
       # Verify index was built
       assert index == mock_index
       mock_index.add.assert_called_once()
       
       # Verify normalization was applied
       mock_normalize.assert_called_once()
