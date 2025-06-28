"""
AI Help Desk - Main Application Entry Point

This module serves as the entry point for the AI Help Desk application.
It integrates all components and provides a FastAPI web interface.
"""
import json
import logging
import os
import time
import uuid
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.classification import RequestClassifier
from src.retrieval import KnowledgeRetriever
from src.response import ResponseGenerator
from src.escalation import EscalationHandler
from src.config import API_TITLE, API_DESCRIPTION, API_VERSION, LOG_LEVEL, LOG_FORMAT

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize components
classifier = RequestClassifier()
retriever = KnowledgeRetriever()
response_generator = ResponseGenerator()
escalation_handler = EscalationHandler()

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models
class HelpDeskRequest(BaseModel):
    request_text: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class HelpDeskResponse(BaseModel):
    response_text: str
    expected_classification: str
    confidence: float
    escalated: bool
    resolution_time: Optional[str] = None

@app.get("/")
async def root():
    """
    Root endpoint that returns basic API information.
    """
    return {
        "name": API_TITLE,
        "description": API_DESCRIPTION,
        "version": API_VERSION,
    }

@app.post("/api/help", response_model=HelpDeskResponse)
async def process_help_request(request: HelpDeskRequest):
    """
    Process a help desk request and generate a response.
    
    Args:
        request: Help desk request containing the user's question
        
    Returns:
        HelpDeskResponse: Generated response and metadata
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Processing request {request_id}: {request.request_text[:50]}...")
        
        # Step 1: Classify the request
        category, confidence, category_details = classifier.classify(request.request_text)
        logger.debug(f"Request {request_id} classified as '{category}' with confidence {confidence:.4f}")
        
        # Step 2: Retrieve relevant knowledge first
        retrieved_docs = retriever.retrieve_by_category(request.request_text, category)
        
        # Step 3: Check if the request should be escalated, passing retrieved docs
        should_escalate, escalation_reason = escalation_handler.should_escalate(
            request.request_text, category, category_details, confidence, retrieved_docs
        )
        
        # Step 4: Generate response
        if should_escalate:
            # Check if we have relevant knowledge base content to include
            has_relevant_content = False
            for doc in retrieved_docs:
                if doc.metadata.get("similarity_score", 0) > 0.3:
                    has_relevant_content = True
                    break
                    
            if has_relevant_content:
                # First generate a response based on the knowledge base
                kb_response = response_generator.generate_response(
                    request.request_text, category, category_details, retrieved_docs, escalated=True
                )
                
                # Then get the escalation message
                escalation_msg = escalation_handler.get_escalation_message(
                    category, escalation_reason
                )
                
                # Combine both responses
                response_text = f"{kb_response}\n\n{escalation_msg}"
                logger.info(f"Providing knowledge base content along with escalation message for {category}")
            else:
                # For cases without relevant content, just use the escalation message
                response_text = escalation_handler.get_escalation_message(
                    category, escalation_reason
                )
        else:
            # Generate normal response
            response_text = response_generator.generate_response(
                request.request_text, category, category_details, retrieved_docs, escalated=False
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = HelpDeskResponse(
            response_text=response_text,
            expected_classification=category,
            confidence=confidence,
            escalated=should_escalate,
            resolution_time=category_details.get("typical_resolution_time")
        )
        
        logger.info(f"Request {request_id} processed in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/api/categories")
async def get_categories():
    """
    Get all available help desk categories.
    
    Returns:
        Dict: Dictionary of categories and their details
    """
    try:
        return {"categories": classifier.categories}
    except Exception as e:
        logger.error(f"Error retrieving categories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving categories: {str(e)}")

@app.post("/api/evaluate")
async def evaluate_test_requests():
    """
    Evaluate the system using test requests.
    
    Returns:
        Dict: Evaluation results
    """
    try:
        # Load test requests
        with open(os.path.join("data", "test_requests.json"), "r") as f:
            test_data = json.load(f)
        
        results = []
        total_requests = len(test_data.get("test_requests", []))
        correct_classifications = 0
        correct_escalations = 0
        
        for test_case in test_data.get("test_requests", []):
            request_text = test_case.get("request")
            expected_classification = test_case.get("expected_classification")
            expected_elements = test_case.get("expected_elements", [])
            expected_escalation = test_case.get("escalate", False)
            
            # Process the request
            category, confidence, category_details = classifier.classify(request_text)
            should_escalate, _ = escalation_handler.should_escalate(
                request_text, category, category_details, confidence
            )
            
            # Check classification accuracy
            classification_correct = category == expected_classification
            if classification_correct:
                correct_classifications += 1
            
            # Check escalation accuracy
            escalation_correct = should_escalate == expected_escalation
            if escalation_correct:
                correct_escalations += 1
            
            # Retrieve and generate response
            retrieved_docs = retriever.retrieve_by_category(request_text, category)
            response_text = response_generator.generate_response(
                request_text, category, category_details, retrieved_docs
            )
            
            # Check for expected elements in response
            elements_found = []
            for element in expected_elements:
                if element.lower() in response_text.lower():
                    elements_found.append(element)
            
            elements_accuracy = len(elements_found) / len(expected_elements) if expected_elements else 1.0
            
            # Add result
            results.append({
                "id": test_case.get("id"),
                "request": request_text,
                "expected_classification": expected_classification,
                "actual_classification": category,
                "classification_correct": classification_correct,
                "expected_escalation": expected_escalation,
                "actual_escalation": should_escalate,
                "escalation_correct": escalation_correct,
                "expected_elements": expected_elements,
                "elements_found": elements_found,
                "elements_accuracy": elements_accuracy
            })
        
        # Calculate overall metrics
        classification_accuracy = correct_classifications / total_requests if total_requests > 0 else 0
        escalation_accuracy = correct_escalations / total_requests if total_requests > 0 else 0
        
        elements_accuracy_sum = sum(r["elements_accuracy"] for r in results)
        avg_elements_accuracy = elements_accuracy_sum / total_requests if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "classification_accuracy": classification_accuracy,
            "escalation_accuracy": escalation_accuracy,
            "response_elements_accuracy": avg_elements_accuracy,
            "overall_accuracy": (classification_accuracy + escalation_accuracy + avg_elements_accuracy) / 3,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error evaluating test requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating test requests: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
