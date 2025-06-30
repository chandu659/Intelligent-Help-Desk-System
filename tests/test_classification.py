"""
Tests for the classification module.
"""
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.classification import RequestClassifier

class TestRequestClassifier(unittest.TestCase):
    """Test cases for the RequestClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock categories file
        self.mock_categories = {
            "categories": {
                "password_reset": {
                    "description": "Password-related issues including resets, lockouts, and policy questions",
                    "typical_resolution_time": "5-10 minutes",
                    "escalation_triggers": ["Multiple failed resets", "Account security concerns"]
                },
                "software_installation": {
                    "description": "Issues with installing, updating, or configuring software applications",
                    "typical_resolution_time": "10-30 minutes",
                    "escalation_triggers": ["Unapproved software requests", "System compatibility issues"]
                }
            }
        }
        
        # Mock the SentenceTransformer
        self.mock_model_patcher = patch('src.classification.SentenceTransformer')
        self.mock_model = self.mock_model_patcher.start()
        self.mock_model_instance = self.mock_model.return_value
        
        # Mock encode method to return predictable embeddings
        def mock_encode(texts):
            if isinstance(texts, list):
                # For category embeddings - return an embedding for each text
                result = []
                for text in texts:
                    if "password" in text.lower():
                        result.append([1.0, 0.0])
                    elif "software" in text.lower():
                        result.append([0.0, 1.0])
                    else:
                        result.append([0.5, 0.5])
                return result
            else:
                # For request embedding
                if "password" in texts.lower():
                    return [0.9, 0.1]
                elif "software" in texts.lower():
                    return [0.1, 0.9]
                else:
                    return [0.5, 0.5]
        
        self.mock_model_instance.encode.side_effect = mock_encode
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.mock_model_patcher.stop()
    
    @patch('builtins.open')
    @patch('json.load')
    def test_load_categories(self, mock_json_load, mock_open):
        """Test loading categories from a file."""
        mock_json_load.return_value = self.mock_categories
        
        classifier = RequestClassifier()
        
        self.assertEqual(len(classifier.categories), 2)
        self.assertIn("password_reset", classifier.categories)
        self.assertIn("software_installation", classifier.categories)
    
    @patch('builtins.open')
    @patch('json.load')
    def test_classify_password_request(self, mock_json_load, mock_open):
        """Test classifying a password reset request."""
        mock_json_load.return_value = self.mock_categories
        
        classifier = RequestClassifier()
        category, confidence, details = classifier.classify("I forgot my password and need to reset it")
        
        self.assertEqual(category, "password_reset")
        self.assertGreater(confidence, 0.5)
    
    @patch('builtins.open')
    @patch('json.load')
    def test_classify_software_request(self, mock_json_load, mock_open):
        """Test classifying a software installation request."""
        mock_json_load.return_value = self.mock_categories
        
        classifier = RequestClassifier()
        category, confidence, details = classifier.classify("I need help installing software on my computer")
        
        self.assertEqual(category, "software_installation")
        self.assertGreater(confidence, 0.5)
    
    @patch('builtins.open')
    @patch('json.load')
    def test_get_escalation_triggers(self, mock_json_load, mock_open):
        """Test getting escalation triggers for a category."""
        mock_json_load.return_value = self.mock_categories
        
        classifier = RequestClassifier()
        triggers = classifier.get_escalation_triggers("password_reset")
        
        self.assertEqual(len(triggers), 2)
        self.assertIn("Multiple failed resets", triggers)
        self.assertIn("Account security concerns", triggers)
    
    @patch('builtins.open')
    @patch('json.load')
    def test_get_resolution_time(self, mock_json_load, mock_open):
        """Test getting resolution time for a category."""
        mock_json_load.return_value = self.mock_categories
        
        classifier = RequestClassifier()
        resolution_time = classifier.get_resolution_time("software_installation")
        
        self.assertEqual(resolution_time, "10-30 minutes")

if __name__ == '__main__':
    unittest.main()
