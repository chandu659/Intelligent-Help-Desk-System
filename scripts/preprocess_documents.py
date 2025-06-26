"""
Preprocessing script for the AI Help Desk application.
Processes raw documents and creates NumPy storage for efficient retrieval.
"""
import os
import sys
import json
import logging
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_DIR, EMBEDDING_MODEL, FAISS_INDEX_PATH,
    KNOWLEDGE_BASE_FILE, COMPANY_POLICIES_FILE,
    TROUBLESHOOTING_DB_FILE, INSTALLATION_GUIDES_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def split_markdown_by_headers(content):
    """
    Split markdown content by headers.
    
    Args:
        content: Markdown content
        
    Returns:
        Dict[str, str]: Dictionary mapping section titles to content
    """
    lines = content.split('\n')
    sections = {}
    current_section = "General"
    current_content = []
    
    for line in lines:
        if line.startswith('# '):
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            
            # Start new section
            current_section = line[2:].strip()
            current_content = []
        elif line.startswith('## '):
            # Save previous section
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            
            # Start new section
            current_section = line[3:].strip()
            current_content = []
        else:
            current_content.append(line)
    
    # Save the last section
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

def chunk_text(text, chunk_size, chunk_overlap):
    """
    Split text into chunks of specified size with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # If not at the end of text, try to find a natural break point
        if end < len(text):
            # Try to find a paragraph break
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2
            else:
                # Try to find a sentence break
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                    end = sentence_break + 2
        
        chunks.append(text[start:end])
        start = end - chunk_overlap
    
    return chunks

def process_markdown_file(file_path, source_type):
    """
    Process a markdown file and extract chunks with metadata.
    
    Args:
        file_path: Path to markdown file
        source_type: Type of source for metadata
        
    Returns:
        List[Dict]: List of document chunks with metadata
    """
    logger.info(f"Processing markdown file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process markdown content by sections
        sections = split_markdown_by_headers(content)
        
        documents = []
        for section_title, section_content in sections.items():
            # Further chunk large sections
            chunks = chunk_text(section_content, CHUNK_SIZE, CHUNK_OVERLAP)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": source_type,
                        "section": section_title,
                        "chunk_id": i
                    }
                })
        
        logger.info(f"Extracted {len(documents)} chunks from {file_path}")
        return documents
    
    except Exception as e:
        logger.error(f"Error processing markdown file {file_path}: {e}")
        return []

def process_troubleshooting_db(file_path):
    """
    Process troubleshooting database JSON file.
    
    Args:
        file_path: Path to troubleshooting database JSON file
        
    Returns:
        List[Dict]: List of document chunks with metadata
    """
    logger.info(f"Processing troubleshooting database: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for issue_key, issue_data in data.get('troubleshooting_steps', {}).items():
            # Create a structured text representation of the troubleshooting steps
            steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(issue_data.get('steps', []))])
            
            text = f"Issue: {issue_key}\nCategory: {issue_data.get('category', '')}\n\nSteps:\n{steps_text}\n\n"
            text += f"Escalation Trigger: {issue_data.get('escalation_trigger', '')}\n"
            text += f"Escalation Contact: {issue_data.get('escalation_contact', '')}"
            
            documents.append({
                "text": text,
                "metadata": {
                    "source": "troubleshooting_db",
                    "issue": issue_key,
                    "category": issue_data.get('category', '')
                }
            })
        
        logger.info(f"Extracted {len(documents)} entries from troubleshooting database")
        return documents
    
    except Exception as e:
        logger.error(f"Error processing troubleshooting database {file_path}: {e}")
        return []

def process_installation_guides(file_path):
    """
    Process installation guides JSON file.
    
    Args:
        file_path: Path to installation guides JSON file
        
    Returns:
        List[Dict]: List of document chunks with metadata
    """
    logger.info(f"Processing installation guides: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for software, guide_data in data.get('software_guides', {}).items():
            # Create a structured text representation of the installation guide
            steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(guide_data.get('steps', []))])
            
            issues_text = ""
            for issue in guide_data.get('common_issues', []):
                issues_text += f"Issue: {issue.get('issue', '')}\n"
                issues_text += f"Solution: {issue.get('solution', '')}\n\n"
            
            text = f"Software: {software}\nTitle: {guide_data.get('title', '')}\n\nInstallation Steps:\n{steps_text}\n\n"
            text += f"Common Issues:\n{issues_text}\n"
            text += f"Support Contact: {guide_data.get('support_contact', '')}"
            
            documents.append({
                "text": text,
                "metadata": {
                    "source": "installation_guides",
                    "software": software
                }
            })
        
        logger.info(f"Extracted {len(documents)} entries from installation guides")
        return documents
    
    except Exception as e:
        logger.error(f"Error processing installation guides {file_path}: {e}")
        return []

def create_faiss_index(documents, model_name, output_path):
    """
    Create a FAISS index from documents.
    
    Args:
        documents: List of document dictionaries with text and metadata
        model_name: Name of the sentence transformer model
        output_path: Path to save the FAISS index and documents
        
    Returns:
        None
    """
    logger.info(f"Creating FAISS index with {len(documents)} documents")
    
    # Extract text from documents
    texts = [doc["text"] for doc in documents]
    
    # Load model and generate embeddings
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info("Generating embeddings")
    embeddings = model.encode(texts)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    logger.info("Creating FAISS index")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save index
    index_file = os.path.join(output_path, "index.faiss")
    logger.info(f"Saving FAISS index to {index_file}")
    faiss.write_index(index, index_file)
    
    # Save documents
    documents_file = os.path.join(output_path, "documents.json")
    logger.info(f"Saving documents to {documents_file}")
    with open(documents_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    logger.info("FAISS index creation complete")

def main():
    parser = argparse.ArgumentParser(description="Preprocess documents for AI Help Desk")
    parser.add_argument("--output", type=str, default=str(FAISS_INDEX_PATH),
                        help="Output directory for FAISS index")
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL,
                        help="Sentence transformer model name")
    args = parser.parse_args()
    
    # Process all document sources
    documents = []
    
    # Process markdown files
    documents.extend(process_markdown_file(KNOWLEDGE_BASE_FILE, "knowledge_base"))
    documents.extend(process_markdown_file(COMPANY_POLICIES_FILE, "company_policies"))
    
    # Process JSON files
    documents.extend(process_troubleshooting_db(TROUBLESHOOTING_DB_FILE))
    documents.extend(process_installation_guides(INSTALLATION_GUIDES_FILE))
    
    # Create FAISS index
    create_faiss_index(documents, args.model, args.output)

if __name__ == "__main__":
    main()
