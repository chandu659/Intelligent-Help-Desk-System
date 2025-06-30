"""
Escalation module for the AI Help Desk application.
Handles determining when requests should be escalated to human agents.
"""
import logging
import re
from typing import Dict, List, Tuple, Any, Optional

from src.config import ESCALATION_THRESHOLD

# Set up logging
logger = logging.getLogger(__name__)

class EscalationHandler:
    """
    Determines when help desk requests should be escalated to human agents.
    Uses category-specific triggers and confidence scores.
    """
    
    def __init__(self, threshold: float = ESCALATION_THRESHOLD):
        """
        Initialize the escalation handler.
        
        Args:
            threshold: Confidence threshold for escalation
        """
        self.threshold = threshold
        logger.info(f"Initialized EscalationHandler with threshold {threshold}")
    
    def should_escalate(
        self,
        request: str,
        category: str,
        category_details: Dict[str, Any],
        confidence: float,
        retrieved_docs: List = None
    ) -> Tuple[bool, str]:
        """
        Determine if a request should be escalated based on retrieved knowledge content, triggers, and confidence scores.
        
        Args:
            request: User request
            category: Classified category
            category_details: Details about the category
            confidence: Classification confidence score
            retrieved_docs: Retrieved knowledge documents
            
        Returns:
            Tuple[bool, str]: Escalation decision and reason
        """
        # First check if we have relevant knowledge base content with good similarity scores
        has_relevant_content = False
        best_similarity = 0.0
        best_doc_source = ""
        best_doc_section = ""
        
        if retrieved_docs:
            for doc in retrieved_docs:
                similarity = doc.metadata.get("similarity_score", 0)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_doc_source = doc.metadata.get("source", "unknown")
                    best_doc_section = doc.metadata.get("section", "")
                    
            # If we have good quality matches, log them
            if best_similarity > 0.3:
                has_relevant_content = True
                logger.info(f"Found relevant content for {category} with similarity score {best_similarity:.4f} from {best_doc_source} - {best_doc_section}")
        
        # Check for category-specific automatic escalation
        # Some categories always require escalation regardless of content
        if category == "security_incident":
            reason = "Security incident requires escalation to security team"
            logger.info(f"Escalating security_incident request to Security Response Team")
            return True, reason
            
        if category == "hardware_failure":
            reason = f"Automatic escalation for category: {category}"
            logger.info(f"Escalating request due to {reason}")
            return True, reason
        
        # Check for escalation triggers in the request
        escalation_triggers = category_details.get("escalation_triggers", [])
        for trigger in escalation_triggers:
            if self._check_trigger_match(request, trigger):
                reason = f"Matched escalation trigger: {trigger}"
                logger.info(f"Escalating request due to {reason}")
                return True, reason
        
        # Check for urgency indicators
        urgency_indicators = [
            "urgent", "urgently", "emergency", "immediately", "asap", "critical",
            "deadline", "tomorrow", "today", "right now", "can't wait"
        ]
        for indicator in urgency_indicators:
            if re.search(r'\b' + re.escape(indicator) + r'\b', request.lower()):
                reason = f"Urgency indicator detected: {indicator}"
                logger.info(f"Escalating request due to {reason}")
                return True, reason
        
        # If we have relevant content with good similarity, use it instead of escalating
        # This applies to all categories, not just specific ones
        if has_relevant_content and best_similarity > 0.3:
            logger.info(f"Not escalating {category} issue despite confidence {confidence:.2f} as we have relevant knowledge base content with similarity {best_similarity:.4f}")
            return False, f"No escalation needed - using knowledge base content from {best_doc_source}"
        
        # Only check confidence threshold last, after all other escalation criteria
        if confidence < self.threshold:
            reason = f"Low classification confidence ({confidence:.2f} < {self.threshold}) and no relevant knowledge base content found"
            logger.info(f"Escalating request due to {reason}")
            return True, reason
        
        # No escalation needed
        logger.debug(f"No escalation needed for request: {request[:50]}...")
        return False, "No escalation needed"
    
    def _check_trigger_match(self, request: str, trigger: str) -> bool:
        """
        Check if a request matches an escalation trigger.
        
        Args:
            request: User request
            trigger: Escalation trigger phrase
            
        Returns:
            bool: True if the request matches the trigger
        """
        # Convert trigger to regex pattern
        # Replace spaces with flexible whitespace matching
        pattern = re.escape(trigger).replace('\\ ', r'\s+')
        
        # Case-insensitive search
        return bool(re.search(pattern, request, re.IGNORECASE))
    
    def get_escalation_message(
        self,
        category: str,
        reason: str,
        estimated_wait_time: str = "15-30 minutes"
    ) -> str:
        """
        Generate an escalation message for the user.
        
        Args:
            category: Request category
            reason: Escalation reason
            estimated_wait_time: Estimated wait time for human response
            
        Returns:
            str: Escalation message
        """
        # Use real data from troubleshooting_database.json
        # These are the actual escalation contacts from the knowledge base
        category_to_contact = {
            "password_reset": "security@techcorp.com",
            "hardware_failure": "hardware-support@techcorp.com",
            "network_connectivity": "network-support@techcorp.com", 
            "email_configuration": "email-support@techcorp.com",
            "software_installation": "software-support@techcorp.com",
            "security_incident": "security@techcorp.com",
            "policy_question": "it-policy@techcorp.com"
        }
        
        # Map categories to team names based on categories.json
        category_to_team = {
            "password_reset": "Account Security Team",
            "hardware_failure": "Hardware Support Team",
            "network_connectivity": "Network Operations Team",
            "email_configuration": "Email Administration Team",
            "software_installation": "Software Support Team",
            "security_incident": "Security Response Team",
            "policy_question": "IT Policy Team"
        }
        
        team_name = category_to_team.get(category, "IT Support Team")
        team_contact = category_to_contact.get(category, "support@techcorp.com")
        
        # Get resolution time from categories.json
        category_resolution_times = {
            "password_reset": "5-10 minutes",
            "software_installation": "10-30 minutes",
            "hardware_failure": "2-3 business days",
            "network_connectivity": "15-45 minutes",
            "email_configuration": "10-20 minutes",
            "security_incident": "Immediate response",
            "policy_question": "5-15 minutes"
        }
        
        resolution_time = category_resolution_times.get(category, estimated_wait_time)
        
        # Log the escalation for debugging
        logger.info(f"Escalating {category} request to {team_name} at {team_contact}")
        
        message = f"""
I'm escalating your request to our {team_name} for specialized assistance.

This is due to: {reason}

What happens next:
1. A support specialist will review your request
2. You should receive a response within {resolution_time}
3. You can contact the team directly at {team_contact}

Thank you for your patience.
"""
        
        return message
        