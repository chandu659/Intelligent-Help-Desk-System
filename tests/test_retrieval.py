"""
Tests for the retrieval module.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import faiss

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval import KnowledgeRetriever, Document

class TestKnowledgeRetriever(unittest.TestCase):
    """Test cases for the KnowledgeRetriever class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the SentenceTransformer
        self.mock_model_patcher = patch('src.retrieval.SentenceTransformer')
        self.mock_model = self.mock_model_patcher.start()
        self.mock_model_instance = self.mock_model.return_value
        
        # Mock faiss
        self.mock_faiss_patcher = patch('src.retrieval.faiss')
        self.mock_faiss = self.mock_faiss_patcher.start()
        
        # Mock faiss index
        self.mock_index = MagicMock()
        self.mock_faiss.IndexFlatIP.return_value = self.mock_index
        
        # Mock file operations
        self.mock_open_patcher = patch('builtins.open', mock_open(read_data="# Test Content\n\nThis is test content."))
        self.mock_file = self.mock_open_patcher.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.mock_model_patcher.stop()
        self.mock_faiss_patcher.stop()
        self.mock_open_patcher.stop()
    
    def test_document_creation(self):
        """Test creating a Document object."""
        doc = Document("Test text", {"source": "test"})
        
        self.assertEqual(doc.text, "Test text")
        self.assertEqual(doc.metadata["source"], "test")
    
    def test_split_markdown_by_headers(self):
        """Test splitting markdown content by headers."""
        content = "# Header 1\nContent 1\n\n## Subheader 1\nSubcontent 1\n\n# Header 2\nContent 2"
        
        retriever = KnowledgeRetriever()
        sections = retriever._split_markdown_by_headers(content)
        
        self.assertEqual(len(sections), 3)
        self.assertIn("Header 1", sections)
        self.assertIn("Subheader 1", sections)
        self.assertIn("Header 2", sections)
        self.assertEqual(sections["Header 1"], "Content 1")
        self.assertEqual(sections["Subheader 1"], "Subcontent 1")
        self.assertEqual(sections["Header 2"], "Content 2")
    
    def test_chunk_text(self):
        """Test chunking text."""
        text = "This is a test. " * 50  # Create text longer than chunk size
        
        retriever = KnowledgeRetriever(chunk_size=100, chunk_overlap=20)
        chunks = retriever._chunk_text(text)
        
        self.assertGreater(len(chunks), 1)
        # Check overlap
        self.assertTrue(chunks[0][-20:] in chunks[1])
    
    @patch('src.retrieval.KnowledgeRetriever._load_markdown_documents')
    @patch('src.retrieval.KnowledgeRetriever._load_troubleshooting_db')
    @patch('src.retrieval.KnowledgeRetriever._load_installation_guides')
    @patch('src.retrieval.KnowledgeRetriever._create_faiss_index')
    def test_initialization(self, mock_create_index, mock_load_guides, mock_load_db, mock_load_md):
        """Test initializing the KnowledgeRetriever."""
        mock_create_index.return_value = (MagicMock(), np.array([]))
        
        retriever = KnowledgeRetriever()
        
        # Check if all loading methods were called
        mock_load_md.assert_called()
        mock_load_db.assert_called()
        mock_load_guides.assert_called()
        mock_create_index.assert_called()
    
    def test_retrieve(self):
        """Test retrieving documents for a query."""
        # Setup mock for faiss search
        self.mock_index.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]]))
        
        # Setup mock for encode
        self.mock_model_instance.encode.return_value = np.array([[1.0, 0.0]])
        
        # Create retriever with mock documents
        retriever = KnowledgeRetriever()
        retriever.documents = [
            Document("Doc 1", {"source": "test"}),
            Document("Doc 2", {"source": "test"}),
            Document("Doc 3", {"source": "test"})
        ]
        retriever.index = self.mock_index
        
        # Test retrieve
        docs = retriever.retrieve("test query")
        
        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0].text, "Doc 1")
        self.assertEqual(docs[1].text, "Doc 2")
        self.assertEqual(docs[2].text, "Doc 3")
    
    def test_retrieve_by_category(self):
        """Test retrieving documents filtered by category."""
        # Setup mock for faiss search
        self.mock_index.search.return_value = (np.array([[0.9, 0.8, 0.7, 0.6, 0.5]]), np.array([[0, 1, 2, 3, 4]]))
        
        # Setup mock for encode
        self.mock_model_instance.encode.return_value = np.array([[1.0, 0.0]])
        
        # Create retriever with mock documents
        retriever = KnowledgeRetriever()
        retriever.documents = [
            Document("Doc 1", {"source": "knowledge_base"}),
            Document("Doc 2", {"source": "company_policies"}),
            Document("Doc 3", {"source": "troubleshooting_db"}),
            Document("Doc 4", {"source": "installation_guides"}),
            Document("Doc 5", {"source": "other"})
        ]
        retriever.index = self.mock_index
        
        # Test retrieve by category
        docs = retriever.retrieve_by_category("test query", "password_reset")
        
        # Should prioritize knowledge_base and company_policies for password_reset
        self.assertEqual(len(docs), 3)  # Default top_k is 3
        self.assertTrue(docs[0].metadata["source"] in ["knowledge_base", "company_policies"])
        self.assertTrue(docs[1].metadata["source"] in ["knowledge_base", "company_policies"])

if __name__ == '__main__':
    unittest.main()
