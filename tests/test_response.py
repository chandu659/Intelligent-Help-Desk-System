"""
Tests for the response generation module.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Document class to avoid importing sentence_transformers
@dataclass
class Document:
    text: str
    metadata: dict

# Patch the retrieval module
sys.modules['src.retrieval'] = MagicMock()
sys.modules['src.retrieval'].Document = Document

# Mock groq module
mock_groq = MagicMock()
mock_groq.Groq = MagicMock()
sys.modules['groq'] = mock_groq

# Now import ResponseGenerator after mocking dependencies
from src.response import ResponseGenerator

class TestResponseGenerator(unittest.TestCase):
    """Test cases for the ResponseGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = ResponseGenerator(model_name="test-model")
        
        # Sample data for testing
        self.request = "I forgot my password and need to reset it"
        self.category = "password_reset"
        self.category_details = {
            "description": "Password-related issues including resets, lockouts, and policy questions",
            "typical_resolution_time": "5-10 minutes",
            "escalation_triggers": ["Multiple failed resets", "Account security concerns"]
        }
        self.retrieved_docs = [
            Document(
                "Passwords must be minimum 8 characters with mixed case, numbers, and symbols. "
                "Password reset can be performed through self-service portal at company.com/reset.",
                {"source": "knowledge_base", "section": "Password Management"}
            ),
            Document(
                "Account lockout occurs after 5 failed login attempts. "
                "Contact IT support if account remains locked after successful reset.",
                {"source": "knowledge_base", "section": "Password Management"}
            )
        ]
    
    def test_create_prompt(self):
        """Test creating a prompt for the LLM."""
        prompt = self.generator._create_prompt(
            self.request, self.category, self.category_details, self.retrieved_docs
        )
        
        # Check if the prompt contains all necessary elements
        self.assertIn(self.request, prompt)
        self.assertIn(self.category, prompt)
        self.assertIn(self.category_details["description"], prompt)
        # Resolution time is no longer included in the prompt
        # self.assertIn(self.category_details["typical_resolution_time"], prompt)
        self.assertIn("knowledge_base", prompt)
        self.assertIn("Password Management", prompt)
        self.assertIn("company.com/reset", prompt)
    
    @patch.object(ResponseGenerator, '_call_llm_api')
    def test_generate_response(self, mock_call_llm):
        """Test generating a response."""
        mock_response = "Here's how to reset your password: Go to company.com/reset and follow the instructions."
        mock_call_llm.return_value = mock_response
        
        response = self.generator.generate_response(
            self.request, self.category, self.category_details, self.retrieved_docs
        )
        
        self.assertEqual(response, mock_response)
        mock_call_llm.assert_called_once()
    
    @patch.object(ResponseGenerator, '_call_llm_api')
    def test_generate_response_with_error(self, mock_call_llm):
        """Test generating a response when the LLM API fails."""
        mock_call_llm.side_effect = Exception("API error")
        
        response = self.generator.generate_response(
            self.request, self.category, self.category_details, self.retrieved_docs
        )
        
        # Should return a fallback response
        self.assertIn("apologize", response.lower())
        self.assertIn(self.category, response.lower())
    
    def test_generate_simulated_response(self):
        """Test generating a simulated response."""
        prompt = self.generator._create_prompt(
            self.request, self.category, self.category_details, self.retrieved_docs
        )
        
        response = self.generator._generate_simulated_response(prompt)
        
        # Check if the response contains relevant information
        self.assertIn("password", response.lower())
        self.assertIn("reset", response.lower())
    
    def test_generate_fallback_response(self):
        """Test generating a fallback response."""
        response = self.generator._generate_fallback_response(self.category)
        
        # Check if the fallback response contains the category
        self.assertIn(self.category, response)
        self.assertIn("apologize", response.lower())

if __name__ == '__main__':
    unittest.main()
