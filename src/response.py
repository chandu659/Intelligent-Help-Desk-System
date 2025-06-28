"""
Response generation module for the AI Help Desk application.
Handles generating helpful responses using LLM and retrieved knowledge.
"""
import logging
import re
from typing import List, Dict, Any

from src.config import DEFAULT_LLM_MODEL, LLM_TEMPERATURE, MAX_TOKENS
from src.retrieval import Document

# Set up logging
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Generates responses to help desk requests using LLM and retrieved knowledge.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ):
        """
        Initialize the response generator.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature parameter for LLM
            max_tokens: Maximum number of tokens in the response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized ResponseGenerator with model {model_name}")
    
    def _create_prompt(
        self,
        request: str,
        category: str,
        category_details: Dict[str, Any],
        retrieved_docs: List[Document],
        escalated: bool = False
    ) -> str:
        """
        Create a prompt for the LLM.
        
        Args:
            request: User request
            category: Classified category
            category_details: Details about the category
            retrieved_docs: Retrieved knowledge documents
            
        Returns:
            str: Formatted prompt for the LLM
        """
        # Format retrieved knowledge
        knowledge_sections = []
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source", "unknown")
            section = doc.metadata.get("section", "")
            source_info = f"{source} - {section}" if section else source
            knowledge_sections.append(f"[Knowledge {i+1} from {source_info}]\n{doc.text}\n")
        
        retrieved_sections = "\n".join(knowledge_sections)
        
        # Format category information
        category_description = category_details.get("description", "")
        resolution_time = category_details.get("typical_resolution_time", "unknown")
        
        # Extract key information from retrieved documents to guide response
        key_topics = set()
        for doc in retrieved_docs:
            # Extract keywords from document text
            text = doc.text.lower()
            
            # Check for common help desk topics in the document
            if any(term in text for term in ["password", "reset", "credential", "login"]):
                key_topics.add("password reset steps")
                key_topics.add("security guidelines")
                
            if any(term in text for term in ["install", "setup", "download", "software"]):
                key_topics.add("installation steps")
                key_topics.add("permissions required")
                
            if any(term in text for term in ["troubleshoot", "hardware", "device", "error"]):
                key_topics.add("troubleshooting steps")
                key_topics.add("hardware support contact")
                
            if any(term in text for term in ["network", "connect", "internet", "wifi"]):
                key_topics.add("network troubleshooting")
                key_topics.add("connection verification")
                
            if any(term in text for term in ["email", "outlook", "mail", "smtp"]):
                key_topics.add("email setup steps")
                key_topics.add("server settings")
                
            if any(term in text for term in ["security", "breach", "virus", "malware"]):
                key_topics.add("security team contact")
                key_topics.add("immediate actions")
                
            if any(term in text for term in ["policy", "compliance", "rule", "guideline"]):
                key_topics.add("relevant policy information")
                key_topics.add("compliance requirements")
        
        # If no specific topics were found, use a generic set based on category
        if not key_topics:
            if category == "password_reset":
                key_topics = {"password reset steps", "security guidelines"}
            elif category == "software_installation":
                key_topics = {"installation steps", "permissions required"}
            elif category == "hardware_failure":
                key_topics = {"troubleshooting steps", "hardware support contact"}
            elif category == "network_connectivity":
                key_topics = {"network troubleshooting", "connection verification"}
            elif category == "email_configuration":
                key_topics = {"email setup steps", "server settings"}
            elif category == "security_incident":
                key_topics = {"security team contact", "immediate actions"}
            elif category == "policy_question":
                key_topics = {"relevant policy information", "compliance requirements"}
            else:
                key_topics = {"relevant troubleshooting steps", "support contact information"}
                
        # Convert set to comma-separated string
        response_elements = ", ".join(key_topics)
        
        # Build the prompt following the implementation guide
        # Add information about whether the issue is escalated
        escalation_instruction = ""
        contact_policy = ""
        
        if escalated:
            escalation_instruction = "\n            9. This is an ESCALATED issue. You MUST include the specific escalation contact email address from the knowledge base (such as network-support@techcorp.com or email-support@techcorp.com) in your response."
            contact_policy = "\n            - For escalated issues, direct users to the specific escalation contact found in the knowledge base."
        else:
            escalation_instruction = "\n            9. This is NOT an escalated issue. You MUST NOT include any contact information or suggest contacting support in your response. Do not mention any support team, help desk, or IT support contacts. Do not tell users to reach out for help or contact anyone."
            contact_policy = "\n            - For non-escalated issues, do not direct users to contact any support teams or individuals."
            
        prompt = f"""You are an IT help desk assistant for TechCorp Inc. Your task is to generate a concise, actionable, and policy-compliant response to a user's help desk request based on the provided request, classified category, and relevant knowledge. Focus only on the information provided and the expected response elements.

            **User Request**: {request}
            **Category**: {category} - {category_description}
            **Relevant Knowledge**:
            {retrieved_sections}
            **Relevant Topics from Knowledge Base**: {response_elements}
            **Issue Escalated**: {escalated}

            **Company Policies**:
            - Only approved software from the IT catalog can be installed.
            - Software installation requires department head approval.
            - Personal software is prohibited on company devices.{contact_policy}

            **Instructions**:
            1. Generate a response that is concise (100-150 words max) and directly addresses the user's request.
            2. Base your answer STRICTLY on the retrieved knowledge and address the relevant topics where applicable: {response_elements}.
            3. Use the relevant knowledge to provide specific, actionable steps or contacts.
            4. DO NOT invent, create, or include ANY information not explicitly mentioned in the retrieved knowledge.
            5. DO NOT include phone numbers, URLs, or contact details that are not explicitly mentioned in the retrieved knowledge.
            6. If the knowledge is insufficient, acknowledge the limitations without making up information.
            7. Ensure the response complies with the listed company policies.
            8. Format the response with a brief introduction, bullet-pointed steps or instructions, and a brief closing statement. DO NOT include resolution time in your response.
            9. Use a professional and friendly tone.
            10. Avoid redundancy - do not repeat the same information multiple times in your response.{escalation_instruction}

            **Response**:"""
        
        return prompt
    
    def generate_response(
        self,
        request: str,
        category: str,
        category_details: Dict[str, Any],
        retrieved_docs: List[Document],
        escalated: bool = False
    ) -> str:
        """
        Generate a response to a help desk request.
        
        Args:
            request: User request
            category: Classified category
            category_details: Details about the category
            retrieved_docs: Retrieved knowledge documents
            escalated: Whether the issue is escalated
            
        Returns:
            str: Generated response
        """
        # Log the request and category
        logger.info(f"Generating response for request: '{request[:50]}...' in category '{category}'")
        
        # Log detailed information about retrieved documents
        logger.info(f"Using {len(retrieved_docs)} retrieved documents for response generation:")
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source", "unknown")
            section = doc.metadata.get("section", "")
            source_info = f"{source} - {section}" if section else source
            similarity = doc.metadata.get("similarity_score", "unknown")
            logger.info(f"Document {i+1}: {source_info} - First 100 chars: {doc.text}...")
        
        # Create prompt for the LLM
        prompt = self._create_prompt(request, category, category_details, retrieved_docs, escalated)
        
        # Call the LLM API using Groq - no fallback responses
        response = self._call_llm_api(prompt)
        
        # Post-process the response to ensure escalation contacts are handled correctly
        response = self._post_process_response(response, escalated, retrieved_docs)
        
        logger.debug(f"Generated response for request: {request[:50]}...")
        return response
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API to generate a response.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            str: Generated response
            
        Raises:
            ImportError: If the Groq package is not available
            Exception: If there's an error calling the Groq API
        """
        # Import groq
        from groq import Groq
        
        # Initialize client - will use GROQ_API from environment variables
        client = Groq()
        
        # Call the Groq API with the model specified in config
        response = client.chat.completions.create(
            model=self.model_name,  # Using the model from config
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        logger.info(f"Successfully generated response using Groq API with model {self.model_name}")
        return response.choices[0].message.content
        
    def _post_process_response(self, response: str, escalated: bool, retrieved_docs: List[Document]) -> str:
        """
        Post-process the response to ensure escalation contacts are handled correctly
        and remove resolution times from all responses.
        
        Args:
            response: The generated response from the LLM
            escalated: Whether the issue is escalated
            retrieved_docs: Retrieved knowledge documents
            
        Returns:
            str: The processed response
        """
        # Start with the original response
        processed_response = response
        
        # Remove any mentions of resolution time for all responses
        resolution_patterns = [
            r'\*\*Resolution Time:?\*\*.*?\n',
            r'Resolution Time:?.*?\n',
            r'Expected Resolution Time:?.*?\n',
            r'Typical Resolution Time:?.*?\n',
            r'Resolution:? \d+-\d+ \w+',
            r'typically resolved in \d+-\d+ \w+',
            r'resolved within \d+-\d+ \w+'
        ]
        
        for pattern in resolution_patterns:
            processed_response = re.sub(pattern, "", processed_response, flags=re.IGNORECASE)
            
        logger.info("Removed resolution times from response")
            
        # If the issue is escalated, we want to keep any escalation contacts
        if escalated:
            return processed_response
            
        # If not escalated, we need to filter out any specific escalation contacts
        # Collect all escalation contacts from the knowledge base
        escalation_contacts = set()
        for doc in retrieved_docs:
            text = doc.text.lower()
            # Extract email addresses that look like escalation contacts
            
            email_pattern = r'\b[\w.-]+\-support@[\w.-]+\.\w+\b'
            matches = re.findall(email_pattern, text)
            for match in matches:
                escalation_contacts.add(match)
                
        # For non-escalated issues, remove all email addresses and references to contacting support
        # Remove all email addresses using regex
        email_pattern = r'\b[\w.-]+@[\w.-]+\.\w+\b'
        processed_response = re.sub(email_pattern, "", processed_response)
        
        # Remove common phrases about contacting support and made-up information
        support_phrases = [
            r"[Pp]lease (?:don't hesitate to |feel free to )?(?:reach out|contact) (?:to )?(?:our|the) (?:general |IT )?support team.*?\.\s*",
            r"[Ff]or (?:any|further) (?:questions|concerns|assistance|help|support).*?(?:reach out|contact|email) (?:to )?(?:our|the) (?:general |IT )?support.*?\.\s*",
            r"[Ii]f you (?:have|need) (?:any|further) (?:questions|concerns|assistance|help|support).*?(?:reach out|contact|email).*?\.\s*",
            r"[Cc]ontact (?:our|the) (?:general |IT )?support team.*?\.\s*",
            r"[Pp]lease (?:contact|reach out to) us.*?\.\s*",
            r"[Dd]on't hesitate to (?:reach out|contact|ask).*?\.\s*"
        ]
        
        # Remove phone numbers that aren't in the knowledge base
        phone_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
        processed_response = re.sub(phone_pattern, "", processed_response)
        
        for phrase in support_phrases:
            processed_response = re.sub(phrase, "", processed_response)
        
        logger.info(f"Removed all email addresses and support contact references from non-escalated response")
        return processed_response
    
   