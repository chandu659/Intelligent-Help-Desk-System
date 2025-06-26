"""
Run script for the AI Help Desk application.
This script initializes the application and starts the FastAPI server.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for running the AI Help Desk application.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the AI Help Desk application")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run preprocessing before starting the server")
    args = parser.parse_args()
    
    # Load environment variables from .env.local
    load_dotenv(".env.local")
    
    # Check if GROQ_API is set
    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY environment variable not set. "
                      "LLM functionality will use fallback responses.")
    
    # Run preprocessing if requested
    if args.preprocess:
        logger.info("Running preprocessing...")
        try:
            from scripts.preprocess_documents import main as preprocess_main
            preprocess_main()
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            sys.exit(1)
    
    # Start the FastAPI server
    logger.info(f"Starting AI Help Desk server on {args.host}:{args.port}")
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
