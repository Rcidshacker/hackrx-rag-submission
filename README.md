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
