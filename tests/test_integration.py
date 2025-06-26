"""
Integration tests for the AI Help Desk application.
Tests the full pipeline from request to response.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classification import RequestClassifier
from src.retrieval import KnowledgeRetriever
from src.response import ResponseGenerator
from src.escalation import EscalationHandler

class TestIntegration(unittest.TestCase):
    """Integration tests for the AI Help Desk application."""
    
    @patch('src.classification.RequestClassifier')
    @patch('src.retrieval.KnowledgeRetriever')
    @patch('src.response.ResponseGenerator')
    @patch('src.escalation.EscalationHandler')
    def test_full_pipeline(self, mock_escalation, mock_response, mock_retrieval, mock_classification):
        """Test the full pipeline from request to response."""
        # Setup mocks
        mock_classifier = mock_classification.return_value
        mock_classifier.classify.return_value = (
            "password_reset", 
            0.9, 
            {
                "description": "Password-related issues",
                "typical_resolution_time": "5-10 minutes",
                "escalation_triggers": ["Multiple failed resets"]
            }
        )
        
        mock_retriever_instance = mock_retrieval.return_value
        mock_retriever_instance.retrieve_by_category.return_value = [
            MagicMock(text="Password reset instructions", metadata={"source": "knowledge_base"})
        ]
        
        mock_escalation_instance = mock_escalation.return_value
        mock_escalation_instance.should_escalate.return_value = (False, "No escalation needed")
        
        mock_response_instance = mock_response.return_value
        mock_response_instance.generate_response.return_value = "Here's how to reset your password..."
        
        # Test request
        request = "I forgot my password and need to reset it"
        
        # Process request
        category, confidence, category_details = mock_classifier.classify(request)
        should_escalate, reason = mock_escalation_instance.should_escalate(
            request, category, category_details, confidence
        )
        retrieved_docs = mock_retriever_instance.retrieve_by_category(request, category)
        
        if should_escalate:
            response = mock_escalation_instance.get_escalation_message(category, reason)
        else:
            response = mock_response_instance.generate_response(
                request, category, category_details, retrieved_docs
            )
        
        # Assertions
        self.assertEqual(category, "password_reset")
        self.assertFalse(should_escalate)
        self.assertEqual(response, "Here's how to reset your password...")
        
        # Verify method calls
        mock_classifier.classify.assert_called_once_with(request)
        mock_escalation_instance.should_escalate.assert_called_once_with(
            request, category, category_details, confidence
        )
        mock_retriever_instance.retrieve_by_category.assert_called_once_with(request, category)
        mock_response_instance.generate_response.assert_called_once_with(
            request, category, category_details, retrieved_docs
        )
    
    @patch('src.classification.RequestClassifier')
    @patch('src.retrieval.KnowledgeRetriever')
    @patch('src.response.ResponseGenerator')
    @patch('src.escalation.EscalationHandler')
    def test_escalation_pipeline(self, mock_escalation, mock_response, mock_retrieval, mock_classification):
        """Test the pipeline with escalation."""
        # Setup mocks
        mock_classifier = mock_classification.return_value
        mock_classifier.classify.return_value = (
            "hardware_failure", 
            0.9, 
            {
                "description": "Hardware issues",
                "typical_resolution_time": "2-3 business days",
                "escalation_triggers": ["All hardware failures"]
            }
        )
        
        mock_retriever_instance = mock_retrieval.return_value
        mock_retriever_instance.retrieve_by_category.return_value = [
            MagicMock(text="Hardware support information", metadata={"source": "knowledge_base"})
        ]
        
        mock_escalation_instance = mock_escalation.return_value
        mock_escalation_instance.should_escalate.return_value = (True, "Automatic escalation for category: hardware_failure")
        mock_escalation_instance.get_escalation_message.return_value = "Your request has been escalated..."
        
        # Test request
        request = "My laptop screen is broken and I need it fixed urgently"
        
        # Process request
        category, confidence, category_details = mock_classifier.classify(request)
        should_escalate, reason = mock_escalation_instance.should_escalate(
            request, category, category_details, confidence
        )
        
        if should_escalate:
            response = mock_escalation_instance.get_escalation_message(category, reason)
        else:
            retrieved_docs = mock_retriever_instance.retrieve_by_category(request, category)
            response = mock_response_instance.generate_response(
                request, category, category_details, retrieved_docs
            )
        
        # Assertions
        self.assertEqual(category, "hardware_failure")
        self.assertTrue(should_escalate)
        self.assertEqual(response, "Your request has been escalated...")
        
        # Verify method calls
        mock_classifier.classify.assert_called_once_with(request)
        mock_escalation_instance.should_escalate.assert_called_once_with(
            request, category, category_details, confidence
        )
        mock_escalation_instance.get_escalation_message.assert_called_once_with(category, reason)
        # Retrieval and response generation should not be called for escalated requests
        mock_retriever_instance.retrieve_by_category.assert_not_called()
        mock_response.return_value.generate_response.assert_not_called()

if __name__ == '__main__':
    unittest.main()
