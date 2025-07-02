, mock_sentence_transformer):
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
   
   @patch('rdb.embedding.models.SentenceTransformer')
   @patch('faiss.IndexFlatIP')
   @patch('faiss.normalize_L2')
   def test_end_to_end_embedding_process(self, mock_normalize, mock_index_class, mock_sentence_transformer):
       """Test end-to-end embedding process with realistic data."""
       # Mock SentenceTransformer
       mock_model = Mock()
       mock_model.get_sentence_embedding_dimension.return_value = 384
       mock_model.max_seq_length = 512
       mock_model.encode.return_value = np.random.rand(3, 384)  # 3 chunks, 384 dimensions
       mock_sentence_transformer.return_value = mock_model
       
       # Mock FAISS index
       mock_index = Mock()
       mock_index.ntotal = 3
       mock_index_class.return_value = mock_index
       
       config = Config()
       embedder = DocumentEmbedder(config)
       
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
