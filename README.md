# AI Help Desk System

An intelligent help desk system that classifies IT support requests, retrieves relevant knowledge, generates contextual responses using LLMs, and determines when to escalate to human agents.

## Project Structure

```
├── data/                    # Input documents and FAISS index
├── src/                     # Source code
│   ├── classification.py    # Request classification module
│   ├── retrieval.py         # Knowledge retrieval module
│   ├── response.py          # Response generation module
│   ├── escalation.py        # Escalation logic module
│   └── config.py            # Configuration settings
├── tests/                   # Unit and integration tests
├── main.py                  # FastAPI application entry point
├── run.py                   # Application runner script
└── requirements.txt         # Python dependencies
```
## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
# Start the server
python main.py

```

## API Endpoints

- `GET /`: Basic API information
- `POST /api/help`: Process a help desk request
- `GET /api/categories`: Get all available categories
- `POST /api/evaluate`: Evaluate system performance

## Testing

```bash
python -m pytest tests/
```
## Technologies

- **Embedding**: SentenceTransformer (all-MiniLM-L6-v2)
- **Vector Search**: FAISS
- **LLM**: Groq's LLama-3.1-8b-instant
- **API Framework**: FastAPI

User request testing from Postman
----------------------------------
POST: http://localhost:8000/api/help

Example Request: RAW (json)
{
  "request_text":"I think someone hacked my computer because I'm seeing strange pop-ups and my browser homepage changed.",
  "user_id": "test_user"
}

Example Response: JSON

{
    "response_text": "**Security Incident Response**\n\nDear [User],\n\nWe have received your report of a potential security threat on your company-issued device. We take all security incidents seriously and will guide you through the next steps.\n\n**Immediate Actions:**\n\n* Do not attempt to fix the issue yourself.\n* Preserve evidence by not deleting any files or resetting your system.\n* Document all steps taken and the timeline of events.\n\n**Security Guidelines:**\n\n* Ensure you are connected to the company VPN for all company resource access.\n* Verify that your device has an encrypted hard drive and company-managed antivirus.\n\n**Security Team Contact:**\n\n* Please report this incident to security@techcorp.com immediately.\n\n**Escalation Contact:**\n\n* If you have any further questions or concerns, please reach out to security@techcorp.com.\n\n**Please feel free to ask if you have any questions.**\n\n\nI'm escalating your request to our Security Response Team for specialized assistance.\n\nThis is due to: Security incident requires escalation to security team\n\nWhat happens next:\n1. A support specialist will review your request\n2. You should receive a response within Immediate response\n3. You can contact the team directly at security@techcorp.com\n\nThank you for your patience.\n",
    "expected_classification": "security_incident",
    "confidence": 0.7498231768608093,
    "escalated": true,
    "resolution_time": "Immediate response"
}