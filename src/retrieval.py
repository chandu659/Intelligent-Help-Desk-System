"""
Retrieval module for the AI Help Desk application.
Handles knowledge retrieval from various data sources.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL, KNOWLEDGE_BASE_FILE, COMPANY_POLICIES_FILE,
    TROUBLESHOOTING_DB_FILE, INSTALLATION_GUIDES_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP
)

# Set up logging
logger = logging.getLogger(__name__)

class Document:
    """
    Represents a document or chunk of text with metadata.
    """
    def __init__(self, text: str, metadata: Dict[str, Any]):
        """
        Initialize a document.
        
        Args:
            text: The document text
            metadata: Document metadata (source, category, etc.)
        """
        self.text = text
        self.metadata = metadata
    
    def __str__(self) -> str:
        return f"Document(source={self.metadata.get('source', 'unknown')}, text={self.text[:50]}...)"

class KnowledgeRetriever:
    """
    Retrieves relevant knowledge from various data sources.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        knowledge_base_file: Path = KNOWLEDGE_BASE_FILE,
        company_policies_file: Path = COMPANY_POLICIES_FILE,
        troubleshooting_db_file: Path = TROUBLESHOOTING_DB_FILE,
        installation_guides_file: Path = INSTALLATION_GUIDES_FILE,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initialize the knowledge retriever.
        
        Args:
            model_name: Name of the sentence transformer model to use
            knowledge_base_file: Path to the knowledge base markdown file
            company_policies_file: Path to the company policies markdown file
            troubleshooting_db_file: Path to the troubleshooting database JSON file
            installation_guides_file: Path to the installation guides JSON file
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between text chunks
        """
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load and process documents
        self.documents = []
        self._load_markdown_documents(knowledge_base_file, "knowledge_base")
        self._load_markdown_documents(company_policies_file, "company_policies")
        self._load_troubleshooting_db(troubleshooting_db_file)
        self._load_installation_guides(installation_guides_file)
        
        # Create FAISS index
        self.index, self.document_embeddings = self._create_faiss_index()
        logger.info(f"Initialized KnowledgeRetriever with {len(self.documents)} documents")
    
    def _load_markdown_documents(self, file_path: Path, source_type: str) -> None:
        """
        Load and chunk markdown documents.
        
        Args:
            file_path: Path to the markdown file
            source_type: Type of source (for metadata)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process markdown content by sections
            sections = self._split_markdown_by_headers(content)
            
            for section_title, section_content in sections.items():
                # Further chunk large sections
                chunks = self._chunk_text(section_content)
                
                for i, chunk in enumerate(chunks):
                    self.documents.append(Document(
                        text=chunk,
                        metadata={
                            "source": source_type,
                            "section": section_title,
                            "chunk_id": i
                        }
                    ))
        except Exception as e:
            logger.error(f"Error loading markdown file {file_path}: {e}")
    
    def _split_markdown_by_headers(self, content: str) -> Dict[str, str]:
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
    
    def _load_troubleshooting_db(self, file_path: Path) -> None:
        """
        Load troubleshooting database from JSON file.
        
        Args:
            file_path: Path to the troubleshooting database JSON file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for issue_key, issue_data in data.get('troubleshooting_steps', {}).items():
                # Create a structured text representation of the troubleshooting steps
                steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(issue_data.get('steps', []))])
                
                text = f"Issue: {issue_key}\nCategory: {issue_data.get('category', '')}\n\nSteps:\n{steps_text}\n\n"
                text += f"Escalation Trigger: {issue_data.get('escalation_trigger', '')}\n"
                text += f"Escalation Contact: {issue_data.get('escalation_contact', '')}"
                
                self.documents.append(Document(
                    text=text,
                    metadata={
                        "source": "troubleshooting_db",
                        "issue": issue_key,
                        "category": issue_data.get('category', '')
                    }
                ))
        except Exception as e:
            logger.error(f"Error loading troubleshooting database {file_path}: {e}")
    
    def _load_installation_guides(self, file_path: Path) -> None:
        """
        Load installation guides from JSON file.
        
        Args:
            file_path: Path to the installation guides JSON file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
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
                
                self.documents.append(Document(
                    text=text,
                    metadata={
                        "source": "installation_guides",
                        "software": software
                    }
                ))
        except Exception as e:
            logger.error(f"Error loading installation guides {file_path}: {e}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of specified size with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # If not at the end of text, try to find a natural break point
            if end < len(text):
                # Try to find a paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Try to find a sentence break
                    sentence_break = text.rfind('. ', start, end)
                    if sentence_break != -1 and sentence_break > start + self.chunk_size // 2:
                        end = sentence_break + 2
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        
        return chunks
    
    def _create_faiss_index(self) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
        """
        Create a FAISS index from document embeddings.
        
        Returns:
            Tuple[faiss.IndexFlatIP, np.ndarray]: FAISS index and document embeddings
        """
        # Extract text from documents
        texts = [doc.text for doc in self.documents]
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        return index, embeddings
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List[Document]: List of retrieved documents
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Get the retrieved documents
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        
        logger.debug(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
        
        return retrieved_docs
    
    def retrieve_by_category(self, query: str, category: str, top_k: int = 3) -> List[Document]:
        """
        Retrieve documents filtered by category.
        
        Args:
            query: Query text
            category: Category to filter by
            top_k: Number of documents to retrieve
            
        Returns:
            List[Document]: List of retrieved documents
        """
        # Map help desk categories to knowledge sources
        category_to_source = {
            "password_reset": ["knowledge_base", "company_policies"],
            "software_installation": ["installation_guides", "troubleshooting_db"],
            "hardware_failure": ["troubleshooting_db", "knowledge_base"],
            "network_connectivity": ["troubleshooting_db", "knowledge_base"],
            "email_configuration": ["troubleshooting_db", "knowledge_base"],
            "security_incident": ["company_policies", "knowledge_base"],
            "policy_question": ["company_policies"]
        }
        
        # Get relevant sources for the category
        relevant_sources = category_to_source.get(category, None)
        
        # If no specific mapping, retrieve from all sources
        if not relevant_sources:
            return self.retrieve(query, top_k)
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, len(self.documents))
        
        # Filter by relevant sources
        filtered_docs = []
        for idx in indices[0]:
            doc = self.documents[idx]
            if doc.metadata.get("source") in relevant_sources:
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break
        
        logger.debug(f"Retrieved {len(filtered_docs)} documents for query: {query[:50]}... in category: {category}")
        
        return filtered_docs
