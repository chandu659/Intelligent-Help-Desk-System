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
        confidence: float
    ) -> Tuple[bool, str]:
        """
        Determine if a request should be escalated based on category, confidence, and triggers.
        
        Args:
            request: User request
            category: Classified category
            category_details: Details about the category
            confidence: Classification confidence score
            
        Returns:
            Tuple[bool, str]: Escalation decision and reason
        """
        # Check if confidence is below threshold
        if confidence < self.threshold:
            reason = f"Low classification confidence ({confidence:.2f} < {self.threshold})"
            logger.info(f"Escalating request due to {reason}")
            return True, reason
        
        # Check category-specific automatic escalation
        if category in ["hardware_failure", "security_incident"]:
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
            "urgent", "emergency", "immediately", "asap", "critical",
            "deadline", "tomorrow", "today", "right now", "can't wait"
        ]
        for indicator in urgency_indicators:
            if re.search(r'\b' + re.escape(indicator) + r'\b', request.lower()):
                reason = f"Urgency indicator detected: {indicator}"
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
        # Map categories to specialized teams and their contact information
        category_to_team = {
            "hardware_failure": {
                "name": "Hardware Support Team",
                "contact": "hardware-support@company.com"
            },
            "security_incident": {
                "name": "Security Response Team",
                "contact": "security-incidents@company.com"
            },
            "network_connectivity": {
                "name": "Network Operations Team",
                "contact": "network-ops@company.com"
            },
            "software_installation": {
                "name": "Software Support Team",
                "contact": "software-support@company.com"
            },
            "email_configuration": {
                "name": "Email Administration Team",
                "contact": "email-admin@company.com"
            },
            "password_reset": {
                "name": "Account Security Team",
                "contact": "account-security@company.com"
            },
            "policy_question": {
                "name": "IT Policy Team",
                "contact": "it-policy@company.com"
            }
        }
        
        team_info = category_to_team.get(category, {"name": "IT Support Team", "contact": "it-support@company.com"})
        team_name = team_info["name"]
        team_contact = team_info["contact"]
        
        # Get self-help resources based on category
        self_help_resources = self._get_self_help_resources(category)
        
        message = f"""
I'm escalating your request to our {team_name} for specialized assistance.

This is due to: {reason}

What happens next:
1. A support specialist will review your request
2. You should receive a response within {estimated_wait_time}
3. You can contact the team directly at {team_contact}

While you wait, you can check these resources for similar issues:
{self_help_resources}

Thank you for your patience.
"""
        
        return message
        
    def _get_self_help_resources(self, category: str) -> str:
        """
        Get self-help resources based on the request category.
        
        Args:
            category: Request category
            
        Returns:
            str: Formatted self-help resources
        """
        # Map categories to self-help resources
        category_resources = {
            "hardware_failure": [
                "Hardware Troubleshooting Guide: https://company.com/kb/hardware",
                "Equipment Request Form: https://company.com/equipment"
            ],
            "security_incident": [
                "Security Incident Reporting Portal: https://company.com/security",
                "Security Best Practices: https://company.com/kb/security-best-practices"
            ],
            "network_connectivity": [
                "Network Status Page: https://company.com/network-status",
                "VPN Setup Guide: https://company.com/kb/vpn-setup"
            ],
            "software_installation": [
                "Software Catalog: https://company.com/software",
                "Installation Guides: https://company.com/kb/software-installation"
            ],
            "email_configuration": [
                "Email Setup Guide: https://company.com/kb/email-setup",
                "Distribution List Management: https://company.com/kb/distribution-lists"
            ],
            "password_reset": [
                "Self-Service Password Reset: https://company.com/reset-password",
                "Account Security Guide: https://company.com/kb/account-security"
            ],
            "policy_question": [
                "IT Policy Portal: https://company.com/policies",
                "Software Installation Policy: https://company.com/kb/software-policy"
            ]
        }
        
        resources = category_resources.get(category, [
            "Knowledge Base: https://company.com/kb",
            "IT Support Portal: https://company.com/support"
        ])
        
        return "\n".join([f"- {resource}" for resource in resources])
