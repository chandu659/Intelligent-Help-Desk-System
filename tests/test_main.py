"""
Tests for the main FastAPI application.
"""
import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main

class TestMainApplication(unittest.TestCase):
    """Test cases for the main FastAPI application."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(main.app)
    
    def test_root_endpoint(self):
        """Test the root endpoint returns correct information."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("name", data)
        self.assertIn("version", data)
        self.assertIn("description", data)
    
    @patch('main.RequestClassifier')
    @patch('main.KnowledgeRetriever')
    @patch('main.ResponseGenerator')
    @patch('main.EscalationHandler')
    def test_help_endpoint(self, mock_escalation, mock_response, mock_retrieval, mock_classification):
        """Test the help endpoint processes requests correctly."""
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
        
        mock_retriever = mock_retrieval.return_value
        mock_retriever.retrieve_by_category.return_value = [
            MagicMock(text="Password reset instructions", metadata={"source": "knowledge_base"})
        ]
        
        mock_escalation_handler = mock_escalation.return_value
        mock_escalation_handler.should_escalate.return_value = (False, "No escalation needed")
        
        mock_response_generator = mock_response.return_value
        mock_response_generator.generate_response.return_value = "Here's how to reset your password..."
        
        # Make request
        response = self.client.post(
            "/api/help",
            json={"request_text": "I forgot my password", "user_id": "user123"}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["category"], "password_reset")
        self.assertEqual(data["response"], "Here's how to reset your password...")
        self.assertFalse(data["escalated"])
        self.assertEqual(data["resolution_time"], "5-10 minutes")
    
    @patch('main.RequestClassifier')
    @patch('main.KnowledgeRetriever')
    @patch('main.ResponseGenerator')
    @patch('main.EscalationHandler')
    def test_help_endpoint_with_escalation(self, mock_escalation, mock_response, mock_retrieval, mock_classification):
        """Test the help endpoint with escalation."""
        # Setup mocks
        mock_classifier = mock_classification.return_value
        mock_classifier.classify.return_value = (
            "hardware_failure", 
            0.9, 
            {
                "description": "Hardware-related issues",
                "typical_resolution_time": "1-2 business days",
                "escalation_triggers": ["All hardware failures"]
            }
        )
        
        mock_escalation_handler = mock_escalation.return_value
        mock_escalation_handler.should_escalate.return_value = (True, "Automatic escalation for category: hardware_failure")
        mock_escalation_handler.get_escalation_message.return_value = "Your request has been escalated to IT Support."
        
        # Make request
        response = self.client.post(
            "/api/help",
            json={"request_text": "My laptop screen is broken", "user_id": "user123"}
        )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["category"], "hardware_failure")
        self.assertEqual(data["response"], "Your request has been escalated to IT Support.")
        self.assertTrue(data["escalated"])
        self.assertEqual(data["resolution_time"], "1-2 business days")
        
        # Verify that retrieval and response generation were not called
        mock_retrieval.return_value.retrieve_by_category.assert_not_called()
        mock_response.return_value.generate_response.assert_not_called()
    
    @patch('main.RequestClassifier')
    def test_categories_endpoint(self, mock_classification):
        """Test the categories endpoint returns all categories."""
        # Setup mock
        mock_classifier = mock_classification.return_value
        mock_classifier.get_all_categories.return_value = {
            "password_reset": {
                "description": "Password-related issues",
                "typical_resolution_time": "5-10 minutes"
            },
            "hardware_failure": {
                "description": "Hardware-related issues",
                "typical_resolution_time": "1-2 business days"
            }
        }
        
        # Make request
        response = self.client.get("/api/categories")
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("categories", data)
        self.assertEqual(len(data["categories"]), 2)
        self.assertIn("password_reset", data["categories"])
        self.assertIn("hardware_failure", data["categories"])
    
    @patch('main.process_test_request')
    def test_evaluate_endpoint(self, mock_process):
        """Test the evaluate endpoint processes test requests correctly."""
        # Setup mock
        mock_process.side_effect = [
            {
                "request_id": "test1",
                "classification_correct": True,
                "escalation_correct": True,
                "response_elements_correct": 0.8,
                "overall_score": 0.9
            },
            {
                "request_id": "test2",
                "classification_correct": False,
                "escalation_correct": True,
                "response_elements_correct": 0.5,
                "overall_score": 0.6
            }
        ]
        
        # Mock the test_requests.json file
        with patch('main.open', unittest.mock.mock_open(read_data=json.dumps([
            {"request_id": "test1", "request_text": "Test 1"},
            {"request_id": "test2", "request_text": "Test 2"}
        ]))):
            # Make request
            response = self.client.post("/api/evaluate")
            
            # Verify response
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("results", data)
            self.assertEqual(len(data["results"]), 2)
            self.assertEqual(data["results"][0]["request_id"], "test1")
            self.assertEqual(data["results"][1]["request_id"], "test2")
            self.assertIn("average_score", data)
            self.assertEqual(data["average_score"], 0.75)  # (0.9 + 0.6) / 2

if __name__ == '__main__':
    unittest.main()
