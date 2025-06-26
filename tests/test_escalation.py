"""
Tests for the escalation module.
"""
import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.escalation import EscalationHandler

class TestEscalationHandler(unittest.TestCase):
    """Test cases for the EscalationHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = EscalationHandler(threshold=0.7)
        
        # Sample data for testing
        self.category_details = {
            "description": "Password-related issues including resets, lockouts, and policy questions",
            "typical_resolution_time": "5-10 minutes",
            "escalation_triggers": ["Multiple failed resets", "Account security concerns"]
        }
    
    def test_should_escalate_low_confidence(self):
        """Test escalation due to low confidence."""
        request = "I forgot my password"
        category = "password_reset"
        confidence = 0.6  # Below threshold
        
        should_escalate, reason = self.handler.should_escalate(
            request, category, self.category_details, confidence
        )
        
        self.assertTrue(should_escalate)
        self.assertIn("Low classification confidence", reason)
    
    def test_should_not_escalate_high_confidence(self):
        """Test no escalation with high confidence."""
        request = "I forgot my password"
        category = "password_reset"
        confidence = 0.8  # Above threshold
        
        should_escalate, reason = self.handler.should_escalate(
            request, category, self.category_details, confidence
        )
        
        self.assertFalse(should_escalate)
        self.assertEqual(reason, "No escalation needed")
    
    def test_should_escalate_automatic_category(self):
        """Test automatic escalation for certain categories."""
        request = "My laptop screen is broken"
        category = "hardware_failure"
        confidence = 0.9  # High confidence
        
        should_escalate, reason = self.handler.should_escalate(
            request, category, self.category_details, confidence
        )
        
        self.assertTrue(should_escalate)
        self.assertIn("Automatic escalation for category", reason)
    
    def test_should_escalate_security_incident(self):
        """Test automatic escalation for security incidents."""
        request = "I think my account has been hacked"
        category = "security_incident"
        confidence = 0.9  # High confidence
        
        should_escalate, reason = self.handler.should_escalate(
            request, category, self.category_details, confidence
        )
        
        self.assertTrue(should_escalate)
        self.assertIn("Automatic escalation for category", reason)
    
    def test_should_escalate_trigger_match(self):
        """Test escalation due to trigger match."""
        request = "I've had multiple failed resets of my password"
        category = "password_reset"
        confidence = 0.8  # High confidence
        
        should_escalate, reason = self.handler.should_escalate(
            request, category, self.category_details, confidence
        )
        
        self.assertTrue(should_escalate)
        self.assertIn("Matched escalation trigger", reason)
    
    def test_should_escalate_urgency_indicator(self):
        """Test escalation due to urgency indicator."""
        request = "I need to reset my password urgently for a meeting"
        category = "password_reset"
        confidence = 0.8  # High confidence
        
        should_escalate, reason = self.handler.should_escalate(
            request, category, self.category_details, confidence
        )
        
        self.assertTrue(should_escalate)
        self.assertIn("Urgency indicator detected", reason)
    
    def test_check_trigger_match(self):
        """Test checking if a request matches a trigger."""
        request = "I've had multiple failed resets of my password"
        trigger = "Multiple failed resets"
        
        result = self.handler._check_trigger_match(request, trigger)
        
        self.assertTrue(result)
    
    def test_check_trigger_no_match(self):
        """Test checking if a request doesn't match a trigger."""
        request = "I need to reset my password"
        trigger = "Multiple failed resets"
        
        result = self.handler._check_trigger_match(request, trigger)
        
        self.assertFalse(result)
    
    def test_get_escalation_message(self):
        """Test generating an escalation message."""
        category = "password_reset"
        reason = "Low confidence"
        
        message = self.handler.get_escalation_message(category, reason)
        
        # Check if the message contains necessary information
        self.assertIn(reason, message)
        self.assertIn("Account Security Team", message)  # Team for password_reset
        self.assertIn("ticket number", message.lower())

if __name__ == '__main__':
    unittest.main()
