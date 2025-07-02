"""
Tests for the retrieval module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rdb.config.settings import Config
from rdb.retrieval.retriever import DocumentRetriever
from rdb.retrieval.index_manager import IndexManager
from rdb.retrieval.refiner import QueryRefiner


class TestIndexManager:
   """Test cases for IndexManager."""
   
   def setup_method(self):
       """Setup test fixtures."""
       self.config = Config()
       self.index_manager = IndexManager(self.config)
   
   @patch('faiss.read_index')
   @patch('pickle.load')
   def test_load_index_success(self, mock_pickle_load, mock_faiss_read, tmp_path):
       """Test successful index loading."""
       # Create mock files
       index_file = tmp_path / "index.faiss"
       metadata_file = tmp_path / "metadata.pkl"
       index_file.touch()
       metadata_file.touch()
       
       # Mock return values
       mock_index = Mock()
       mock_index.ntotal = 100
       mock_faiss_read.return_value = mock_index
       
       mock_chunks = [{'test': 'chunk'}] * 100
       mock_pickle_load.return_value = mock_chunks
       
       # Test loading
       result = self.index_manager.load_index(str(tmp_path))
       
       assert result is True
       assert self.index_manager.index == mock_index
       assert self.index_manager.chunks == mock_chunks
       assert len(self.index_manager.chunks) == 100
   
   def test_load_index_missing_files(self, tmp_path):
       """Test loading index with missing files."""
       result = self.index_manager.load_index(str(tmp_path))
       assert result is False
   
   @patch('faiss.write_index')
   @patch('pickle.dump')
   def test_save_index(self, mock_pickle_dump, mock_faiss_write, tmp_path):
       """Test saving index and metadata."""
       mock_index = Mock()
       test_chunks = [{'test': 'chunk'}]
       
       index_file, metadata_file = self.index_manager.save_index(
           mock_index, test_chunks, str(tmp_path)
       )
       
       # Verify files were "saved"
       assert "index.faiss" in index_file
       assert "metadata.pkl" in metadata_file
       
       # Verify external libraries were called
       mock_faiss_write.assert_called_once()
       mock_pickle_dump.assert_called_once()
   
   def test_search_not_loaded(self):
       """Test searching when index is not loaded."""
       query_embedding = np.array([[0.1, 0.2, 0.3]])
       
       with pytest.raises(RuntimeError, match="Index not loaded"):
           self.index_manager.search(query_embedding, 5)
   
   def test_search_loaded(self):
       """Test searching when index is loaded."""
       # Mock loaded index
       mock_index = Mock()
       mock_index.search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))
       
       self.index_manager.index = mock_index
       self.index_manager.chunks = [{'chunk': 1}, {'chunk': 2}]
       
       query_embedding = np.array([[0.1, 0.2, 0.3]])
       scores, indices = self.index_manager.search(query_embedding, 2)
       
       mock_index.search.assert_called_once_with(query_embedding, 2)
       assert scores.shape == (1, 2)
       assert indices.shape == (1, 2)
   
   def test_get_stats_not_loaded(self):
       """Test getting stats when index is not loaded."""
       stats = self.index_manager.get_stats()
       assert stats["status"] == "not_loaded"
   
   def test_get_stats_loaded(self):
       """Test getting stats when index is loaded."""
       # Mock loaded index and chunks
       mock_index = Mock()
       mock_index.ntotal = 150
       mock_index.d = 384
       
       test_chunks = [
           {'chunk_type': 'small'},
           {'chunk_type': 'medium'},
           {'chunk_type': 'small'},
           {'chunk_type': 'large'}
       ]
       
       self.index_manager.index = mock_index
       self.index_manager.chunks = test_chunks
       
       stats = self.index_manager.get_stats()
       
       assert stats["status"] == "loaded"
       assert stats["total_vectors"] == 150
       assert stats["total_chunks"] == 4
       assert stats["vector_dimension"] == 384
       assert stats["chunk_types"]["small"] == 2
       assert stats["chunk_types"]["medium"] == 1
       assert stats["chunk_types"]["large"] == 1


class TestDocumentRetriever:
   """Test cases for DocumentRetriever."""
   
   def setup_method(self):
       """Setup test fixtures."""
       self.config = Config()
       self.config.enable_query_refinement = False  # Disable for basic tests
       
       with patch('rdb.retrieval.retriever.EmbeddingModel'), \
            patch('rdb.retrieval.retriever.IndexManager'):
           self.retriever = DocumentRetriever(self.config)
   
   @patch('rdb.retrieval.retriever.IndexManager')
   def test_load_index(self, mock_index_manager_class):
       """Test loading index through retriever."""
       mock_index_manager = Mock()
       mock_index_manager.load_index.return_value = True
       mock_index_manager_class.return_value = mock_index_manager
       
       retriever = DocumentRetriever(self.config)
       result = retriever.load_index()
       
       assert result is True
       mock_index_manager.load_index.assert_called_once()
   
   def test_search_index_not_loaded(self):
       """Test search when index is not loaded."""
       # Mock index manager to return False for is_loaded
       self.retriever.index_manager.is_loaded.return_value = False
       self.retriever.index_manager.load_index.return_value = False
       
       with pytest.raises(RuntimeError, match="Index not loaded"):
           self.retriever.search("test query")
   
   def test_search_success(self):
       """Test successful search."""
       # Mock index manager
       self.retriever.index_manager.is_loaded.return_value = True
       self.retriever.index_manager.search.return_value = (
           np.array([[0.9, 0.8, 0.7]]),  # scores
           np.array([[0, 1, 2]])         # indices
       )
       self.retriever.index_manager.chunks = [
           {
               'page_title': 'Test Page 1',
               'section_path': 'Section 1',
               'url': 'http://example.com/test1',
               'content': 'Test content 1',
               'chunk_type': 'medium',
               'section_level': 2
           },
           {
               'page_title': 'Test Page 2',
               'section_path': 'Section 2', 
               'url': 'http://example.com/test2',
               'content': 'Test content 2',
               'chunk_type': 'small',
               'section_level': 3
           },
           {
               'page_title': 'Test Page 3',
               'section_path': 'Section 3',
               'url': 'http://example.com/test3', 
               'content': 'Test content 3',
               'chunk_type': 'large',
               'section_level': 1
           }
       ]
       
       # Mock embedding model
       self.retriever.embedding_model.encode_query.return_value = np.array([0.1, 0.2, 0.3])
       
       results = self.retriever.search("test query", top_k=3)
       
       assert len(results) == 3
       assert results[0]['rank'] == 1
       assert results[0]['score'] == 0.9
       assert results[0]['page_title'] == 'Test Page 1'
       assert results[1]['rank'] == 2
       assert results[2]['rank'] == 3
   
   @patch('rdb.retrieval.retriever.QueryRefiner')
   def test_search_with_query_refinement(self, mock_refiner_class):
       """Test search with query refinement enabled."""
       # Enable query refinement
       self.config.enable_query_refinement = True
       
       # Mock query refiner
       mock_refiner = Mock()
       mock_refiner.refine_query.return_value = "enhanced test query"
       mock_refiner_class.return_value = mock_refiner
       
       # Create new retriever with refinement enabled
       with patch('rdb.retrieval.retriever.EmbeddingModel'), \
            patch('rdb.retrieval.retriever.IndexManager'):
           retriever = DocumentRetriever(self.config)
           retriever.query_refiner = mock_refiner
       
       # Mock index manager
       retriever.index_manager.is_loaded.return_value = True
       retriever.index_manager.search.return_value = (
           np.array([[0.9]]), np.array([[0]])
       )
       retriever.index_manager.chunks = [
           {
               'page_title': 'Test Page',
               'section_path': 'Section',
               'url': 'http://example.com/test',
               'content': 'Test content',
               'chunk_type': 'medium',
               'section_level': 2
           }
       ]
       
       # Mock embedding model
       retriever.embedding_model.encode_query.return_value = np.array([0.1, 0.2, 0.3])
       
       results = retriever.search("test query", refine_query=True)
       
       # Verify query refinement was used
       mock_refiner.refine_query.assert_called_once_with("test query")
       assert results[0]['original_query'] == "test query"
       assert results[0]['final_query'] == "enhanced test query"


class TestQueryRefiner:
   """Test cases for QueryRefiner."""
   
   @patch('rdb.retrieval.refiner.AutoTokenizer')
   @patch('rdb.retrieval.refiner.AutoModelForCausalLM')
   @patch('torch.cuda.is_available')
   def test_init_success(self, mock_cuda_available, mock_model_class, mock_tokenizer_class):
       """Test successful QueryRefiner initialization."""
       mock_cuda_available.return_value = False
       
       # Mock tokenizer
       mock_tokenizer = Mock()
       mock_tokenizer.pad_token = None
       mock_tokenizer.eos_token = '<eos>'
       mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
       
       # Mock model
       mock_model = Mock()
       mock_model_class.from_pretrained.return_value = mock_model
       
       config = Config()
       config.refiner_model = "test-model-path"
       
       refiner = QueryRefiner(config)
       
       assert refiner.model_path == "test-model-path"
       assert refiner.device == "cpu"
       assert refiner.tokenizer == mock_tokenizer
       assert refiner.model == mock_model
   
   def test_init_no_model(self):
       """Test QueryRefiner initialization with no model available."""
       config = Config()
       config.refiner_model = None
       
       with patch('os.path.exists', return_value=False):
           with pytest.raises(ValueError, match="No refiner model found"):
               QueryRefiner(config)
   
   @patch('rdb.retrieval.refiner.AutoTokenizer')
   @patch('rdb.retrieval.refiner.AutoModelForCausalLM')
   @patch('torch.cuda.is_available')
   def test_refine_query(self, mock_cuda_available, mock_model_class, mock_tokenizer_class):
       """Test query refinement."""
       mock_cuda_available.return_value = False
       
       # Mock tokenizer
       mock_tokenizer = Mock()
       mock_tokenizer.pad_token = None
       mock_tokenizer.eos_token = '<eos>'
       mock_tokenizer.eos_token_id = 2
       mock_tokenizer.return_value = {
           'input_ids': [[1, 2, 3]],
           'attention_mask': [[1, 1, 1]]
       }
       mock_tokenizer.decode.return_value = "prompt text refined: networking configuration troubleshooting"
       mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
       
       # Mock model
       mock_model = Mock()
       mock_outputs = Mock()
       mock_outputs.__getitem__.return_value = [1, 2, 3, 4, 5]  # Mock output tokens
       mock_model.generate.return_value = mock_outputs
       mock_model_class.from_pretrained.return_value = mock_model
       
       config = Config()
       config.refiner_model = "test-model"
       config.refiner_max_tokens = 30
       config.refiner_temperature = 0.7
       
       refiner = QueryRefiner(config)
       
       result = refiner.refine_query("wifi broken")
       
       # Should return the refined part (after the prompt)
       assert "networking configuration troubleshooting" in result
       
       # Verify model.generate was called with correct parameters
       mock_model.generate.assert_called_once()
       call_kwargs = mock_model.generate.call_args[1]
       assert call_kwargs['max_new_tokens'] == 30
       assert call_kwargs['temperature'] == 0.7


class TestRetrievalIntegration:
   """Integration tests for the retrieval module."""
   
   def test_end_to_end_retrieval_workflow(self, tmp_path):
       """Test complete retrieval workflow."""
       config = Config(data_dir=str(tmp_path))
       config.enable_query_refinement = False  # Disable for simpler test
       
       # Create mock index files
       index_file = tmp_path / "index" / "index.faiss"
       metadata_file = tmp_path / "index" / "metadata.pkl"
       index_file.parent.mkdir(parents=True)
       
       # Mock test chunks
       test_chunks = [
           {
               'page_title': 'NetworkManager',
               'section_path': 'Configuration',
               'content': 'Configure NetworkManager for wireless connections',
               'url': 'https://wiki.archlinux.org/title/NetworkManager',
               'chunk_type': 'medium',
               'section_level': 2
           },
           {
               'page_title': 'Wireless Configuration',
               'section_path': 'iwctl',
               'content': 'Use iwctl to manage wireless connections',
               'url': 'https://wiki.archlinux.org/title/Iwd',
               'chunk_type': 'small', 
               'section_level': 3
           }
       ]
       
       # Mock all the external dependencies
       with patch('faiss.read_index') as mock_read_index, \
            patch('pickle.load') as mock_pickle_load, \
            patch('rdb.retrieval.retriever.EmbeddingModel') as mock_embedding_class, \
            patch('faiss.normalize_L2'):
           
           # Mock FAISS index
           mock_index = Mock()
           mock_index.ntotal = 2
           mock_index.d = 384
           mock_index.search.return_value = (
               np.array([[0.95, 0.87]]),  # High similarity scores
               np.array([[0, 1]])         # Indices pointing to our test chunks
           )
           mock_read_index.return_value = mock_index
           
           # Mock pickle load
           mock_pickle_load.return_value = test_chunks
           
           # Mock embedding model
           mock_embedding_model = Mock()
           mock_embedding_model.encode_query.return_value = np.array([0.1, 0.2, 0.3])
           mock_embedding_class.return_value = mock_embedding_model
           
           # Create index files (empty, but they need to exist)
           index_file.touch()
           metadata_file.touch()
           
           # Test the retrieval
           retriever = DocumentRetriever(config)
           
           # Load index
           load_success = retriever.load_index()
           assert load_success is True
           
           # Search
           results = retriever.search("wireless network setup", top_k=2)
           
           # Verify results
           assert len(results) == 2
           assert results[0]['page_title'] == 'NetworkManager'
           assert results[0]['score'] == 0.95
           assert results[1]['page_title'] == 'Wireless Configuration'
           assert results[1]['score'] == 0.87
           
           # Verify both results contain networking-related content
           assert 'wireless' in results[0]['content'].lower() or 'network' in results[0]['content'].lower()
           assert 'wireless' in results[1]['content'].lower() or 'iwctl' in results[1]['content'].lower()
