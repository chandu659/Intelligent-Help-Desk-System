"""
Response generation module for the AI Help Desk application.
Handles generating helpful responses using LLM and retrieved knowledge.
"""
import logging
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
        retrieved_docs: List[Document]
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
        
        knowledge_text = "\n".join(knowledge_sections)
        
        # Format category information
        category_description = category_details.get("description", "")
        resolution_time = category_details.get("typical_resolution_time", "unknown")
        
        # Get escalation contacts if available
        escalation_contacts = category_details.get("escalation_contacts", [])
        contact_info = "\n".join([f"- {contact}" for contact in escalation_contacts]) if escalation_contacts else "No specific contacts available"
        
        # Build the prompt
        prompt = f"""You are an AI-powered IT help desk assistant for a corporate environment. Respond to the user's request professionally and helpfully.

        USER REQUEST: {request}

        CATEGORY: {category} - {category_description}
        TYPICAL RESOLUTION TIME: {resolution_time}
        ESCALATION CONTACTS: {contact_info}

        RELEVANT KNOWLEDGE:
        {knowledge_text}

        INSTRUCTIONS:
        1. Use ONLY the provided knowledge to answer the user's question accurately.
        2. Be professional, clear, and concise in your response.
        3. Include specific steps or solutions when applicable.
        4. DO NOT make up information that is not in the provided knowledge.
        5. DO NOT include fake phone numbers, ticket IDs, or other made-up contact information.
        6. If escalation is needed, refer to actual department emails or contact methods from the knowledge base.
        7. Format your response appropriately with sections and bullet points as needed.
        8. If the knowledge base doesn't contain enough information, acknowledge this and suggest contacting the appropriate team rather than making up a solution.

        YOUR RESPONSE:"""
        
        return prompt
    
    def generate_response(
        self,
        request: str,
        category: str,
        category_details: Dict[str, Any],
        retrieved_docs: List[Document]
    ) -> str:
        """
        Generate a response to a help desk request.
        
        Args:
            request: User request
            category: Classified category
            category_details: Details about the category
            retrieved_docs: Retrieved knowledge documents
            
        Returns:
            str: Generated response
        """
        # Create the prompt
        prompt = self._create_prompt(request, category, category_details, retrieved_docs)
        
        try:
            # In a production environment, this would call the actual LLM API
            # For now, we'll use a placeholder implementation
            response = self._call_llm_api(prompt)
            logger.debug(f"Generated response for request: {request[:50]}...")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback response
            return self._generate_fallback_response(category)
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API to generate a response.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            str: Generated response
        """
        # Use Groq API as specified in the README
        try:
            # Import groq
            from groq import Groq
            
            # Initialize client - will use GROQ_API from environment variables
            client = Groq()
            
            # Call the Groq API with llama-3.1-8b-instant model as specified in README
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Using the model specified in README
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            logger.info(f"Successfully generated response using Groq API")
            return response.choices[0].message.content
        except ImportError:
            logger.warning("Groq package not available, using fallback response")
            return self._generate_simulated_response(prompt)
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            return self._generate_simulated_response(prompt)
    
    # def _generate_simulated_response(self, prompt: str) -> str:
    #     """
    #     Generate a simulated response based on the prompt.
    #     This is used when the actual LLM API is not available.
        
    #     Args:
    #         prompt: Formatted prompt
            
    #     Returns:
    #         str: Simulated response
    #     """
    #     # Extract the category from the prompt
    #     category_line = [line for line in prompt.split('\n') if line.startswith("CATEGORY:")][0]
    #     category = category_line.split("-")[0].replace("CATEGORY:", "").strip()
        
    #     # Generate a response based on the category
    #     responses = {
    #         "password_reset": """
    #     I can help you reset your password. Here's what you need to do:

    #     1. Go to the self-service portal at company.com/reset
    #     2. Enter your company email address
    #     3. Check your email for a reset link (including spam folder)
    #     4. Create a new password following our policy (minimum 8 characters with mixed case, numbers, and symbols)
    #     5. Update your password in any saved locations like browsers or mobile apps

    #     This type of issue typically takes 5-10 minutes to resolve. If you continue to experience issues after resetting your password, please let me know.
    #     """,
    #                 "software_installation": """
    #     I understand you're having trouble installing software. Here's how to resolve this:

    #     1. Make sure you have administrator privileges on your computer
    #     2. Temporarily disable your antivirus during installation
    #     3. Clear your browser cache and temp files if the download seems corrupted
    #     4. Try running the installer as administrator (right-click, "Run as administrator")
    #     5. If the software isn't on the approved list, you'll need manager approval

    #     This type of issue typically takes 10-30 minutes to resolve. If you continue to have problems after trying these steps, please provide the specific error message you're seeing.
    #     """,
    #                 "hardware_failure": """
    #     I'm sorry to hear about your hardware issue. Hardware problems require attention from our support team:

    #     1. Please report this immediately to hardware-support@techcorp.com
    #     2. Back up any important data if possible
    #     3. Our team will assess the issue and may provide temporary equipment for critical needs

    #     Hardware issues typically take 2-3 business days to resolve completely. Given the nature of hardware failures, this issue will be escalated to our hardware support team who will contact you shortly.
    #     """,
    #                 "network_connectivity": """
    #     I can help you troubleshoot your network connectivity issue:

    #     1. First, check your physical cable connections
    #     2. Try restarting your network adapter in Device Manager
    #     3. Connect to the TechCorp-Guest network to test if it's a hardware or network issue
    #     4. Update your network drivers if the connection is unstable
    #     5. If using VPN, check the server status page

    #     Network connectivity issues typically take 15-45 minutes to resolve. If these steps don't resolve your issue, please provide more details about your specific network setup.
    #     """,
    #                 "email_configuration": """
    #     I can help you with your email configuration issue:

    #     1. Check your internet connectivity first
    #     2. Verify Outlook is not in offline mode
    #     3. Try the Send/Receive All Folders option
    #     4. Check your account settings:
    #     - IMAP: mail.company.com, port 993, SSL enabled
    #     - SMTP: smtp.company.com, port 587, TLS enabled
    #     5. If needed, disable and re-enable your email account

    #     Email configuration issues typically take 10-20 minutes to resolve. If these steps don't work, you may need to create a new Outlook profile.
    #     """,
    #                 "security_incident": """
    #     Thank you for reporting this potential security incident. This requires immediate attention:

    #     1. DO NOT attempt to fix this yourself
    #     2. Disconnect your computer from the network immediately
    #     3. Report this incident to security@techcorp.com right away
    #     4. Document what happened and any unusual behavior you noticed
    #     5. Our security team will contact you promptly

    #     Security incidents require immediate response and will be escalated to our security team. They will guide you through the next steps to ensure company data remains protected.
    #     """,
    #                 "policy_question": """
    #     Regarding your policy question:

    #     According to our company IT policies:
    #     - Only approved software from the IT catalog can be installed
    #     - Business software requires department head approval
    #     - Development tools require CTO approval
    #     - All software must pass security review
    #     - Personal software is prohibited on company devices

    #     Policy questions typically take 5-15 minutes to address. If you need clarification or have a specific exception request, please let me know.
    #     """
    #     }
        
    #     # Return the appropriate response or a generic one
    #     return responses.get(category, """
    #         I'll help you with your request. Based on the information you've provided, here are some steps you can take:

    #         1. Check our knowledge base for similar issues
    #         2. Follow the troubleshooting steps provided
    #         3. If the issue persists, please provide more details

    #         Our team typically resolves these types of issues within 1-2 business days. Is there anything specific about your situation I should know to provide better assistance?
    #         """)
    
    # def _generate_fallback_response(self, category: str) -> str:
    #     """
    #     Generate a fallback response when the LLM API fails.
        
    #     Args:
    #         category: Request category
            
    #     Returns:
    #         str: Fallback response
    #     """
    #     return f"""
    #     I apologize, but I'm having trouble generating a specific response for your {category} request right now.

    #     Please try one of the following:
    #     1. Check our self-service portal for immediate assistance
    #     2. Email it-support@techcorp.com with your request

    #     Our team will get back to you as soon as possible. Thank you for your patience.
    #     """
