Project : Intelligent Help Desk System
Technologies: Python, LLM API, Vector Search, REST API
Scenario
Build an AI-powered help desk system that can classify user requests, provide relevant solutions from a knowledge base, and route complex issues appropriately.
Requirements
Create a system with the following components:
1.	Request Classification: Categorize incoming requests into predefined types
2.	Knowledge Retrieval: Search relevant information from provided documents
3.	Response Generation: Generate contextual responses using an LLM
4.	Escalation Logic: Determine when human intervention is needed
Provided Assets
∙	Sample help desk documents (FAQ, procedures, policies)
∙	20 test user requests with expected classifications
Evaluation Criteria
# AI Help Desk System

## Evaluation Criteria

- Architecture (25%): Clean separation of concerns, extensible design
- AI Integration (25%): Effective LLM usage and prompt engineering
- Retrieval Quality (20%): Relevant information extraction
- Code Quality (20%): Testing, error handling, documentation
- Performance Analysis (10%): Evaluation methodology and insights


intelligent-helpdesk/
├── data/                    # Input documents and FAISS index
├── scripts/                 # Preprocessing and utility scripts
│   └── preprocess_documents.py
├── src/                     # Source code
│   ├── classification.py    # Request classification module
│   ├── retrieval.py         # Knowledge retrieval module
│   ├── response.py          # Response generation module
│   ├── escalation.py        # Escalation logic module
│   └── config.py            # Configuration and environment variables
├── tests/                   # Unit and integration tests
├── main.py                  # FastAPI application entry point
├── run.py                   # Application runner script
├── requirements.txt         # Python dependencies
├── .env.local               # Environment variables (not tracked)
├── .env.sample              # Sample environment variables template
└── README.md                # This file

## System Overview

This repository contains an AI-powered help desk system designed to classify user requests, retrieve relevant information from a knowledge base, generate contextual responses using a Large Language Model (LLM), and route complex issues for human escalation. The system is built using Python, leverages `llama-3.1-8b-instant` for LLM capabilities via Groq API, FAISS for vector search, and FastAPI for RESTful integration.

### Key Components

1. **Request Classification**: Categorizes incoming help desk requests using semantic similarity
2. **Knowledge Retrieval**: Finds relevant information from various knowledge sources using vector search
3. **Response Generation**: Creates helpful, contextual responses using LLM technology
4. **Escalation Logic**: Determines when requests should be handled by human agents

## Installation

1. Clone the repository
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.sample` to `.env.local` and fill in your Groq API key:
   ```bash
   cp .env.sample .env.local
   # Edit .env.local with your preferred text editor
   ```

## Usage

### Preprocessing

Before running the application, preprocess the knowledge base documents to create the FAISS index:

```bash
python run.py --preprocess
```

Or run the preprocessing script directly:

```bash
python scripts/preprocess_documents.py
```

### Running the Application

Start the FastAPI server:

```bash
python run.py
```

For development with auto-reload:

```bash
python run.py --reload
```

### API Endpoints

- `GET /`: Basic API information
- `POST /api/help`: Process a help desk request
- `GET /api/categories`: Get all available help desk categories
- `POST /api/evaluate`: Evaluate the system using test requests

### Example API Request

```bash
curl -X POST "http://localhost:8000/api/help" \
     -H "Content-Type: application/json" \
     -d '{"request_text": "I forgot my password and need to reset it", "user_id": "user123"}'
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run specific tests:

```bash
python -m pytest tests/test_classification.py
python -m pytest tests/test_retrieval.py
python -m pytest tests/test_response.py
python -m pytest tests/test_escalation.py
python -m pytest tests/test_integration.py
```

## LLM Integration

The system uses Groq API to access the `llama-3.1-8b-instant` model. Example usage:

```python
from groq import Groq
client = Groq()
completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "Explain why fast inference is critical for reasoning models"
        }
    ]
)
print(completion.choices[0].message.content)
```

## Performance Evaluation

The system includes built-in evaluation capabilities. Run the evaluation endpoint to assess:

- Classification accuracy
- Escalation accuracy
- Response quality
- Overall system performance

```bash
curl -X POST "http://localhost:8000/api/evaluate"
```

Based on the provided documents and the ML Engineer Assessment instructions, the goal is to design an AI-powered help desk system that classifies user requests, retrieves relevant information from a knowledge base, generates contextual responses using a Large Language Model (LLM), and determines when escalation to human intervention is needed. Below is a detailed plan and design instructions, including justifications for each component, to meet the requirements outlined in the assessment.

This design provides a robust, modular, and AI-driven help desk system that meets the assessment requirements. It leverages LLMs for classification and response generation, vector search for efficient knowledge retrieval, and rule-based logic for escalation, all integrated via a REST API. The plan is optimized for the 4-hour time limit, with clear justifications aligning with the evaluation criteria: architecture (modular design), AI integration (LLM and vector search), retrieval quality (Sentence-BERT and FAISS), code quality (testing and documentation), and performance analysis (metrics-driven evaluation).

This repository contains an AI-powered help desk system designed to classify user requests, retrieve relevant information from a knowledge base, generate contextual responses using a Large Language Model (LLM), and route complex issues for human escalation. The system is built using Python, leverages llama-3.1-8b-instant" for LLM capabilities, FAISS for vector search, and FastAPI for RESTful integration.

GROQ_API is a env variable created in env.local  use this 
exmaple of chat completion. 
from groq import Groq
client = Groq()
completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "Explain why fast inference is critical for reasoning models"
        }
    ]
)
print(completion.choices[0].message.content)

---

### **Project Plan and Design Instructions**

#### **1. System Overview**
The intelligent help desk system will process incoming user requests, classify them into predefined categories, retrieve relevant information from provided documents, generate appropriate responses, and flag requests requiring human escalation. The system will leverage Python, an LLM API (e.g., Grok 3), vector search for knowledge retrieval, and a REST API for integration.

**Key Components**:
1. **Request Classification**: Categorize user requests using a machine learning model or LLM-based classification.
2. **Knowledge Retrieval**: Use vector search to extract relevant information from provided documents.
3. **Response Generation**: Generate human-like responses using an LLM with tailored prompts.
4. **Escalation Logic**: Identify requests requiring human intervention based on predefined triggers.

**Evaluation Alignment**:
- **Architecture (25%)**: Modular design with clear separation of concerns.
- **AI Integration (25%)**: Effective use of LLM for classification and response generation.
- **Retrieval Quality (20%)**: Accurate and relevant information retrieval.
- **Code Quality (20%)**: Robust error handling, testing, and documentation.
- **Performance Analysis (10%)**: Metrics to evaluate classification accuracy and response relevance.

---

#### **2. System Architecture**

**Architecture Design**:
The system will follow a modular, pipeline-based architecture to ensure extensibility and maintainability. Each component will be encapsulated as a separate module, communicating through well-defined interfaces.

**Components and Flow**:
1. **Input Processing**:
   - **Function**: Parse and preprocess incoming user requests (text).
   - **Implementation**: Use Python's `re` or `nltk` for text cleaning (e.g., removing special characters, normalizing case).
   - **Justification**: Clean input ensures accurate classification and retrieval by removing noise.

2. **Request Classification Module**:
   - **Function**: Classify requests into categories (e.g., password_reset, hardware_failure) based on `categories.json` and `test_requests.json`.
   - **Implementation**:
     - Use an LLM (e.g., Grok 3) for zero-shot or few-shot classification by providing examples from `sample_conversations.json` and `test_requests.json`.
     - Alternatively, fine-tune a lightweight model (e.g., BERT or DistilBERT) on the provided test requests for better performance on low-resource environments.
     - Input: Preprocessed user request.
     - Output: Category label (e.g., "password_reset").
   - **Justification**:
     - LLM-based classification is flexible and can handle diverse requests without extensive labeled data.
     - Fine-tuned models offer better performance for repetitive tasks but require more setup time.
     - The provided `test_requests.json` and `sample_conversations.json` contain labeled examples, making supervised learning feasible.

3. **Knowledge Retrieval Module**:
   - **Function**: Retrieve relevant information from documents (`installation_guides.json`, `company_it_policies.md`, `knowledge_base.md`, `troubleshooting_database.json`).
   - **Implementation**:
     - Use a vector search engine (e.g., FAISS or Elasticsearch with sentence embeddings).
     - Convert documents into embeddings using a model like Sentence-BERT (`sentence-transformers`).
     - Index document sections (e.g., troubleshooting steps, policy sections) for efficient retrieval.
     - Query the index with the user request to retrieve top-k relevant sections.
   - **Justification**:
     - Vector search ensures semantic relevance, capturing meaning beyond keyword matching.
     - Sentence-BERT is efficient for embedding short document sections, aligning with the structured data in provided files.
     - The documents contain specific, actionable information (e.g., troubleshooting steps, policies), making them ideal for retrieval.

4. **Response Generation Module**:
   - **Function**: Generate a contextual, user-friendly response using retrieved information and an LLM.
   - **Implementation**:
     - Design a prompt template that includes:
       - User request.
       - Classified category.
       - Retrieved document sections.
       - Instructions to generate a concise, actionable response including expected elements (e.g., from `expected_response_elements` in `sample_conversations.json`).
     - Use Grok 3 API to generate the response.
     - Post-process the response to ensure clarity and adherence to company policies (e.g., from `company_it_policies.md`).
   - **Justification**:
     - LLMs excel at generating natural language responses, ensuring user-friendly communication.
     - Prompt engineering with retrieved context ensures responses are specific and policy-compliant.
     - Post-processing mitigates potential LLM inaccuracies (e.g., hallucinations).

5. **Escalation Logic Module**:
   - **Function**: Determine if a request requires human intervention based on `escalation_required` and `escalation_triggers` from `categories.json` and `sample_conversations.json`.
   - **Implementation**:
     - Rule-based logic: Check if the classified category matches escalation triggers (e.g., "All hardware failures require escalation" for `hardware_failure`).
     - For ambiguous cases, use LLM to assess urgency or complexity (e.g., mentions of deadlines or security concerns).
     - Output: Boolean (escalate or not) and escalation contact (e.g., `security@techcorp.com`).
   - **Justification**:
     - Rule-based logic ensures consistency with predefined policies.
     - LLM-based assessment handles edge cases where urgency is implied but not explicit (e.g., "due tomorrow" in `conv_003`).

6. **REST API Layer**:
   - **Function**: Expose the system as a REST API for integration with external platforms.
   - **Implementation**:
     - Use FastAPI or Flask to create endpoints:
       - `POST /process_request`: Accepts user request, returns category, response, and escalation status.
       - `GET /health`: Checks system status.
     - Input: JSON with user request (e.g., `{"request": "I forgot my password"}`).
     - Output: JSON with category, response, escalation status, and contact (if applicable).
   - **Justification**:
     - REST API ensures compatibility with web or mobile frontends, aligning with modern help desk systems.
     - FastAPI offers async support and automatic OpenAPI documentation, improving developer experience.

**Architecture Diagram**:
```
[User Request] --> [Input Processing] --> [Request Classification]
                                                    |
                                                [Knowledge Retrieval]
                                                    |
                                                [Response Generation]
                                                    |
                                                [Escalation Logic]
                                                    |
                                                  [REST API]
                                                    |
                                                [Response to User]
```

**Justification**:
- **Modularity**: Each component is independent, allowing easy updates (e.g., swapping LLM providers).
- **Scalability**: Vector search and REST API support high request volumes.
- **Extensibility**: New categories or documents can be added without redesigning the system.

---

#### **3. Implementation Plan**

**Step 1: Setup and Dependencies**
- **Tools**: Python 3.9+, FastAPI, Sentence-BERT (`sentence-transformers`), FAISS, Grok 3 API (or equivalent LLM), `pandas` for data handling.
- **Data Preparation**:
  - Load and preprocess documents (`installation_guides.json`, etc.) into a unified format (e.g., list of text chunks with metadata).
  - Create a vector index for all document sections using Sentence-BERT and FAISS.
- **Justification**: Standardizing data and precomputing embeddings reduces runtime latency.

**Step 2: Request Classification**
- **Approach**:
  - Use Grok 3 for zero-shot classification with a prompt like:
    ```
    Classify the following user request into one of these categories: [list categories from categories.json].
    Request: {user_request}
    Provide the category name and confidence score.
    ```
  - Alternatively, train a DistilBERT model using `test_requests.json` and `sample_conversations.json` (20+ examples are sufficient for fine-tuning).
- **Justification**: Zero-shot LLM is faster to implement; fine-tuned DistilBERT may improve accuracy for specific categories.

**Step 3: Knowledge Retrieval**
- **Approach**:
  - Embed document sections using Sentence-BERT.
  - Store embeddings in FAISS with metadata (e.g., document source, section title).
  - Query with user request to retrieve top-3 relevant sections.
- **Justification**: Top-k retrieval ensures comprehensive context for response generation.

**Step 4: Response Generation**
- **Approach**:
  - Prompt template:
    ```
    You are an IT help desk assistant for TechCorp Inc. Based on the user request, classified category, and provided information, generate a concise, actionable response. Include specific steps or contacts if relevant, and ensure compliance with company policies.

    User Request: {user_request}
    Category: {category}
    Relevant Information: {retrieved_sections}
    Expected Response Elements: {expected_elements}
    ```
  - Use Grok 3 API to generate the response.
  - Validate response for policy compliance (e.g., no personal software allowed per `company_it_policies.md`).
- **Justification**: Structured prompts ensure responses are relevant and policy-compliant.

**Step 5: Escalation Logic**
- **Approach**:
  - Check `escalation_required` from `sample_conversations.json` or `escalation_triggers` from `categories.json`.
  - For dynamic cases, use LLM to detect urgency keywords (e.g., "tomorrow," "security").
  - Return escalation status and contact (e.g., `security@techcorp.com` for `security_incident`).
- **Justification**: Combining rule-based and LLM-based logic balances reliability and flexibility.

**Step 6: REST API**
- **Approach**:
  - Implement FastAPI endpoints:
    ```python
    from fastapi import FastAPI
    app = FastAPI()

    @app.post("/process_request")
    async def process_request(request: dict):
        user_request = request["request"]
        category = classify_request(user_request)
        retrieved_info = retrieve_knowledge(user_request, category)
        response = generate_response(user_request, category, retrieved_info)
        escalation = check_escalation(category, user_request)
        return {
            "category": category,
            "response": response,
            "escalation_required": escalation["required"],
            "escalation_contact": escalation.get("contact", None)
        }
    ```
- **Justification**: FastAPI is lightweight and supports async operations, ideal for real-time request processing.

**Step 7: Testing and Validation**
- **Approach**:
  - Use `test_requests.json` to evaluate classification accuracy (e.g., precision, recall).
  - Validate response quality by checking inclusion of `expected_elements`.
  - Test escalation logic against `escalation_required` fields.
  - Use unit tests (e.g., `pytest`) for each module.
- **Justification**: Comprehensive testing ensures reliability and aligns with the 20% code quality criterion.

**Step 8: Performance Analysis**
- **Metrics**:
  - **Classification Accuracy**: Percentage of correctly classified requests.
  - **Retrieval Precision**: Proportion of retrieved sections relevant to the request.
  - **Response Quality**: Manual scoring of response completeness and policy compliance.
  - **Latency**: Time taken to process a request (target < 2 seconds).
- **Approach**:
  - Run the system on `test_requests.json` and compute metrics.
  - Log edge cases (e.g., ambiguous requests) for future improvements.
- **Justification**: Quantitative metrics and qualitative analysis meet the 10% performance analysis criterion.

---

#### **4. Code Quality Considerations**

- **Documentation**:
  - Include docstrings for all functions and modules.
  - Provide a README with setup instructions and API usage.
- **Error Handling**:
  - Handle LLM API failures (e.g., timeouts, rate limits).
  - Validate input requests for emptiness or invalid characters.
  - Gracefully handle missing document sections or invalid categories.
- **Testing**:
  - Unit tests for classification, retrieval, and escalation logic.
  - Integration tests for the full pipeline using `test_requests.json`.
- **Justification**: Robust code quality ensures maintainability and reliability, addressing the 20% evaluation criterion.

---

#### **5. Development Timeline (4 Hours)**

- **Hour 1**: Setup environment, load and preprocess documents, initialize FAISS index.
- **Hour 2**: Implement request classification (LLM-based or fine-tuned model).
- **Hour 3**: Build knowledge retrieval and response generation modules.
- **Hour 4**: Implement escalation logic, REST API, and testing; perform performance analysis.

**Justification**: The timeline prioritizes critical components (classification, retrieval) while reserving time for testing and refinement, fitting the 4-hour constraint.

---

#### **6. Sample Workflow for a Request**

**Example Request**: "I forgot my password and can't log into my computer. How do I reset it?" (`req_001`)

1. **Classification**:
   - Category: `password_reset` (matches `test_requests.json`).
2. **Retrieval**:
   - Retrieved: `troubleshooting_database.json` (password_reset steps), `company_it_policies.md` (password policy).
3. **Response Generation**:
   - Prompt: Includes request, category, retrieved steps (e.g., "Go to https://password.techcorp.com"), and expected elements (e.g., "self-service portal").
   - Response: 
     ```
     To reset your password, please visit https://password.techcorp.com and enter your company email address. Check your email (including spam folder) for a reset link and create a new password with at least 12 characters, including uppercase, lowercase, numbers, and symbols. Update the password in all saved locations and test your login. If you encounter issues, contact security@techcorp.com.
     ```
4. **Escalation**:
   - Escalate: False (per `test_requests.json`).
   - Contact: None.

**Justification**: This workflow demonstrates accurate classification, relevant retrieval, policy-compliant response, and correct escalation handling.

---

#### **7. Potential Challenges and Mitigations**

- **Challenge**: LLM generates incorrect or non-compliant responses.
  - **Mitigation**: Strict prompt engineering and post-processing to enforce policy adherence.
- **Challenge**: Vector search retrieves irrelevant sections.
  - **Mitigation**: Fine-tune embedding model or use hybrid search (keyword + vector).
- **Challenge**: Limited time for fine-tuning classification model.
  - **Mitigation**: Use zero-shot LLM classification as a fallback, leveraging provided examples.

---






