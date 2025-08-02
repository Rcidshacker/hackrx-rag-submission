# 🚀 HackRx 6.0: Intelligent Query-Retrieval System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green)
![Architecture](https://img.shields.io/badge/Architecture-RAG-orange)

A complete, production-ready **Retrieval-Augmented Generation (RAG)** system, purpose-built for the HackRx 6.0 competition. This system can ingest large unstructured policy documents (PDFs), understand natural language questions, and generate **fact-based, explainable answers** by citing specific clauses from the source documents.

> Built as a stateless FastAPI web service, this solution is ready for seamless deployment on modern cloud platforms.

---

## ✨ Key Features

- **📄 On-the-Fly Document Processing**  
  Downloads and processes documents from a URL during each API request — ensuring stateless, scalable behavior.

- **🧹 Modular Text Cleaning**  
  Uses a configurable YAML (`config/cleaning_patterns.yaml`) to clean headers, footers, and document noise with regex.

- **📑 Clause-Level Chunking**  
  Leverages NLTK to split clean text into fine-grained, semantically meaningful clauses for precise retrieval.

- **🧠 Semantic Search with FAISS**  
  Employs `all-mpnet-base-v2` sentence transformer for embedding, and **FAISS** for fast, similarity-based clause retrieval.

- **🧾 Fact-Based Answer Generation**  
  Uses a powerful LLM via **OpenRouter**, instructed to answer strictly from the retrieved clauses — ensuring explainability and minimizing hallucination.

- **🧪 API-First Design**  
  Built with **FastAPI**, supporting bearer token authentication, strict Pydantic validation, and hackathon-compliant endpoints.

---

## 🏗️ RAG Workflow: System Architecture

1. **Ingest & Process** (Triggered via API):
   - **Download** the PDF from a given URL
   - **Extract** raw text using PyMuPDF
   - **Clean** it with regex rules from YAML config
   - **Chunk** into semantically meaningful clauses
   - **Embed & Index** in a temporary FAISS vector store

2. **Retrieve:**
   - Embed the query using `all-mpnet-base-v2`
   - Find top 5 semantically similar clauses from the FAISS index

3. **Augment:**
   - Combine the top retrieved clauses with the user query into a custom prompt

4. **Generate:**
   - Send the prompt to the LLM via OpenRouter
   - Return a final answer grounded *only* in the retrieved clauses

---

## 🧰 Tech Stack

| Category             | Tool/Library                     |
|----------------------|----------------------------------|
| Backend Framework    | FastAPI                          |
| Web Server           | Uvicorn, Gunicorn (production)   |
| Document Processing  | PyMuPDF                          |
| Semantic Search      | SentenceTransformers, FAISS      |
| Text Chunking        | NLTK                             |
| LLM Integration      | OpenRouter                       |
| Config Management    | PyYAML, python-dotenv            |

---

## 🧪 Local Setup & Testing

### 1️⃣ Prerequisites

- Python 3.11
- Git

### 2️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/Rcidshacker/hackrx-rag-submission.git
cd hackrx-rag-submission

# Create and activate a virtual environment
py -3.11 -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

3️⃣ Configuration

Create a .env file in the root directory:
# .env
OPENROUTER_API_KEY="sk-or-v1-YourSecretApiKeyHere"
OPENROUTER_SITE_URL="http://localhost:8000"

4️⃣ Running End-to-End Local Test
Terminal 1: Start the FastAPI Server
uvicorn api:app --reload

Terminal 2: Serve PDFs via Local HTTP Server
cd input_docs
python -m http.server 8080

Terminal 3: Run the Test Script
python test_submission.py

✅ This will simulate the full pipeline — from document download to LLM answer generation.

☁️ Deploying to Render
Step-by-Step Guide:

    Push Code to GitHub:
    Ensure the latest version is in a public GitHub repo.

    Create a New Web Service on Render:

        Connect GitHub account

        Select the repo

    Set Configuration:

        Environment: Python 3

        Build Command:
pip install -r requirements.txt

Start Command:
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app

Add Environment Variables:

    PYTHON_VERSION: 3.11

    OPENROUTER_API_KEY: sk-or-v1-YourSecretApiKeyHere

    OPENROUTER_SITE_URL: https://your-app-name.onrender.com

Deploy.
After deployment, your webhook endpoint becomes:
https://your-app-name.onrender.com/api/v1/hackrx/run

📂 Folder Structure
hackrx-rag-submission/
├── api.py                   # FastAPI main server
├── test_submission.py       # Local test script
├── config/
│   └── cleaning_patterns.yaml
├── input_docs/              # Sample PDF directory
├── requirements.txt
├── .env                     # Local API keys (not committed)

📜 License

This project is open-sourced for HackRx 6.0 evaluation and educational use.

🙌 Acknowledgements

    HackRx 6.0 Team for the problem statement

    Hugging Face & OpenRouter for incredible tools

---

Let me know if you'd like to:

- Add contributors section
- Add usage examples via `curl` or Postman
- Include diagrams (I can generate a Mermaid flowchart or architecture PNG)

Ready to go.
