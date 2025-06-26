"""
Configuration module for the AI Help Desk application.
Contains environment variables, paths, and settings.
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"

# Data file paths
CATEGORIES_FILE = DATA_DIR / "categories.json"
KNOWLEDGE_BASE_FILE = DATA_DIR / "knowledge_base.md"
COMPANY_POLICIES_FILE = DATA_DIR / "company_it_policies.md"
TROUBLESHOOTING_DB_FILE = DATA_DIR / "troubleshooting_database.json"
INSTALLATION_GUIDES_FILE = DATA_DIR / "installation_guides.json"
TEST_REQUESTS_FILE = DATA_DIR / "test_requests.json"
SAMPLE_CONVERSATIONS_FILE = DATA_DIR / "sample_conversations.json"

# FAISS index settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# API settings
API_TITLE = "AI Help Desk API"
API_DESCRIPTION = "AI-powered help desk for IT support"
API_VERSION = "0.1.0"

# LLM settings
DEFAULT_LLM_MODEL = "llama3-70b-8192"  # Using Groq as provider based on requirements.txt
LLM_TEMPERATURE = 0.2
MAX_TOKENS = 1024

# Escalation settings
ESCALATION_THRESHOLD = 0.7  # Confidence threshold for escalation

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_settings() -> Dict[str, Any]:
    """
    Returns all configuration settings as a dictionary.
    
    Returns:
        Dict[str, Any]: Configuration settings
    """
    return {
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "embedding_model": EMBEDDING_MODEL,
        "faiss_index_path": FAISS_INDEX_PATH,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "llm_model": DEFAULT_LLM_MODEL,
        "llm_temperature": LLM_TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "log_level": LOG_LEVEL,
        "log_format": LOG_FORMAT
    }
