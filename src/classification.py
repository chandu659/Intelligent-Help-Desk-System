"""
Classification module for the AI Help Desk application.
Handles request classification to determine the appropriate category.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CATEGORIES_FILE, EMBEDDING_MODEL

# Set up logging
logger = logging.getLogger(__name__)

class RequestClassifier:
    """
    Classifies help desk requests into predefined categories.
    Uses sentence embeddings to match requests to the most relevant category.
    """
    
    def __init__(self, categories_file: Path = CATEGORIES_FILE, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the request classifier.
        
        Args:
            categories_file: Path to the categories JSON file
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.categories = self._load_categories(categories_file)
        self.category_embeddings = self._create_category_embeddings()
        logger.info(f"Initialized RequestClassifier with {len(self.categories)} categories")
    
    def _load_categories(self, categories_file: Path) -> Dict:
        """
        Load categories from JSON file.
        
        Args:
            categories_file: Path to the categories JSON file
            
        Returns:
            Dict: Dictionary of categories
        """
        try:
            with open(categories_file, 'r') as f:
                data = json.load(f)
                return data.get('categories', {})
        except Exception as e:
            logger.error(f"Error loading categories file: {e}")
            raise
    
    def _create_category_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Create embeddings for each category based on description and enhanced keywords.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping category names to their embeddings
        """
        # Enhanced keywords for each category to improve classification
        category_enhancements = {
            "password_reset": "forgot password reset login credentials cannot access account locked out",
            "software_installation": "install update upgrade software application program permission error admin rights",
            "hardware_failure": "broken hardware device laptop desktop computer monitor screen keyboard mouse printer not working failure",
            "network_connectivity": "internet connection wifi network down slow cannot connect access website vpn ethernet",
            "email_configuration": "email outlook exchange gmail not receiving messages setup configure distribution list group contacts",
            "security_incident": "virus malware hack suspicious phishing spam unauthorized access security breach alert warning",
            "policy_question": "policy rule guideline allowed permitted forbidden regulation compliance what is the policy can I install personal"
        }
        
        category_texts = {}
        for category, details in self.categories.items():
            # Create a rich text representation of the category with enhanced keywords
            enhancements = category_enhancements.get(category, "")
            text = f"{category} {details.get('description', '')} {enhancements}"
            category_texts[category] = text
        
        # Generate embeddings for all category texts
        texts = list(category_texts.values())
        embeddings = self.model.encode(texts)
        
        # Map category names to their embeddings
        return {cat: embeddings[i] for i, cat in enumerate(category_texts.keys())}
    
    def classify(self, request: str, threshold: float = 0.5) -> Tuple[str, float, Dict]:
        """
        Classify a help desk request into the most relevant category.
        
        Args:
            request: The help desk request text
            threshold: Confidence threshold for classification
            
        Returns:
            Tuple[str, float, Dict]: Category name, confidence score, and category details
        """
        # Preprocess the request to enhance classification
        enhanced_request = self._preprocess_request(request)
        
        # Generate embedding for the request
        request_embedding = self.model.encode(enhanced_request)
        
        # Calculate similarity scores with each category
        scores = {}
        for category, embedding in self.category_embeddings.items():
            # Cosine similarity
            similarity = np.dot(request_embedding, embedding) / (
                np.linalg.norm(request_embedding) * np.linalg.norm(embedding)
            )
            scores[category] = float(similarity)
        
        # Apply keyword-based boosting for specific cases
        scores = self._apply_keyword_boosting(request, scores)
        
        # Find the category with the highest score
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]
        
        logger.debug(f"Request classified as '{best_category}' with confidence {confidence:.4f}")
        logger.debug(f"All scores: {scores}")
        
        # Return the best category, confidence score, and category details
        return best_category, confidence, self.categories.get(best_category, {})
        
    def _preprocess_request(self, request: str) -> str:
        """
        Preprocess the request to enhance classification accuracy.
        
        Args:
            request: The original help desk request text
            
        Returns:
            str: Enhanced request text
        """
        # Convert to lowercase for better matching
        request_lower = request.lower()
        
        # Add additional context based on keywords in the request
        enhancements = []
        
        # Email-related keywords
        if any(word in request_lower for word in ['email', 'outlook', 'gmail', 'message', 'inbox', 'distribution list']):
            enhancements.append("email configuration setup distribution list")
            
        # Policy-related keywords
        if any(word in request_lower for word in ['policy', 'allowed', 'permitted', 'rule', 'can i', 'what is the']):
            enhancements.append("policy question company guidelines rules")
            
        # Software-related keywords
        if any(word in request_lower for word in ['install', 'software', 'application', 'program', 'permission']):
            enhancements.append("software installation permission rights")
            
        # Hardware-related keywords
        if any(word in request_lower for word in ['laptop', 'computer', 'screen', 'keyboard', 'mouse', 'printer', 'broken']):
            enhancements.append("hardware failure device not working")
            
        # Network-related keywords
        if any(word in request_lower for word in ['wifi', 'internet', 'connect', 'network', 'access', 'website']):
            enhancements.append("network connectivity internet access")
            
        # Password-related keywords
        if any(word in request_lower for word in ['password', 'login', 'forgot', 'reset', 'account', 'locked']):
            enhancements.append("password reset account locked")
            
        # Security-related keywords
        if any(word in request_lower for word in ['virus', 'malware', 'hack', 'suspicious', 'security', 'breach']):
            enhancements.append("security incident breach malware")
            
        # Combine the original request with enhancements
        enhanced_request = request
        if enhancements:
            enhanced_request = f"{request} {' '.join(enhancements)}"
            
        return enhanced_request
        
    def _apply_keyword_boosting(self, request: str, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply keyword-based boosting to the similarity scores.
        
        Args:
            request: The original help desk request text
            scores: Dictionary of category scores
            
        Returns:
            Dict[str, float]: Updated scores with boosting applied
        """
        request_lower = request.lower()
        
        # Define keyword boosts for specific categories
        keyword_boosts = {  
            "email_configuration": [
                (0.15, ['distribution list', 'email', 'outlook', 'exchange', 'gmail', 'message', 'inbox'])
            ],
            "policy_question": [
                (0.15, ['policy', 'allowed', 'permitted', 'rule', 'what is the policy', 'can i'])
            ],
            "software_installation": [
                (0.1, ['install', 'software', 'application', 'program', 'permission'])
            ],
            "hardware_failure": [
                (0.1, ['broken', 'laptop', 'computer', 'screen', 'keyboard', 'mouse', 'printer'])
            ],
            "network_connectivity": [
                (0.1, ['wifi', 'internet', 'connect', 'network', 'access', 'website'])
            ],
            "password_reset": [
                (0.1, ['password', 'login', 'forgot', 'reset', 'account', 'locked'])
            ],
            "security_incident": [
                (0.15, ['virus', 'malware', 'hack', 'suspicious', 'security', 'breach'])
            ]
        }
        
        # Apply boosts based on keywords
        for category, boost_rules in keyword_boosts.items():
            # Only apply boost if the category exists in scores
            if category in scores:
                for boost_value, keywords in boost_rules:
                    if any(keyword in request_lower for keyword in keywords):
                        scores[category] += boost_value
                    
        return scores
    
    def get_escalation_triggers(self, category: str) -> List[str]:
        """
        Get escalation triggers for a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List[str]: List of escalation triggers for the category
        """
        if category in self.categories:
            return self.categories[category].get('escalation_triggers', [])
        return []
    
    def get_resolution_time(self, category: str) -> Optional[str]:
        """
        Get typical resolution time for a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Optional[str]: Typical resolution time for the category
        """
        if category in self.categories:
            return self.categories[category].get('typical_resolution_time')
        return None
