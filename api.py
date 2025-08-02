# api.py (Corrected)

import os
import logging
import requests
import tempfile
import json
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict

# Import our project modules
from src.file_handler import extract_text_from_pdf
from src.text_cleaner import load_cleaning_patterns, clean_text_with_patterns, post_process_text
from src.clause_chunker import chunk_text_into_clauses
from src.embedding_generator import generate_and_save_embeddings
from src.llm_handler import get_answer_from_llm
from retriever import SemanticSearcher

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="HackRx Intelligent Query-Retrieval System",
    description="API for processing documents and answering questions using RAG.",
    version="1.0.0"
)
security = HTTPBearer()

EXPECTED_TOKEN = "a0f73b66d6d32b7707a37b571356dd469eadc2f5091c18ac75dd77ceee634a4c"
os.makedirs('output_clauses', exist_ok=True)
os.makedirs('output_embeddings', exist_ok=True)

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

class LocalQueryRequest(BaseModel):
    document_name: str
    question: str

class LocalQueryResponse(BaseModel):
    answer: str
    retrieved_clauses: List[Dict[str, str]]

# --- Helper Functions ---
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authentication token")

def process_document_on_the_fly(doc_url: str) -> str:
    tmp_file_path = None  # Initialize to avoid unbound variable warning
    try:
        logging.info(f"Downloading document from URL: {doc_url}")
        response = requests.get(doc_url, timeout=60)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        base_name = os.path.splitext(os.path.basename(tmp_file_path))[0]
        logging.info(f"Document saved to temporary file: {tmp_file_path}")
        
        raw_text = extract_text_from_pdf(tmp_file_path)
        if not raw_text: raise ValueError("Failed to extract text.")
        
        cleaning_patterns = load_cleaning_patterns('config/cleaning_patterns.yaml')
        cleaned_text = post_process_text(clean_text_with_patterns(raw_text, cleaning_patterns))
        
        clauses = chunk_text_into_clauses(cleaned_text, f"{base_name}.pdf")
        clauses_path = os.path.join('output_clauses', f"{base_name}_clauses.json")
        with open(clauses_path, 'w', encoding='utf-8') as f: json.dump(clauses, f, indent=4)
        
        index_path = os.path.join('output_embeddings', f"{base_name}.index")
        generate_and_save_embeddings(clauses, index_path)
        
        return base_name
    except Exception as e:
        logging.error(f"Error during document processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# --- API Endpoints ---
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest, _=Security(verify_token)):
    document_base_name = process_document_on_the_fly(request.documents)
    searcher = SemanticSearcher(document_base_name)
    if not searcher.index or not searcher.clauses:
        raise HTTPException(status_code=500, detail="Searcher initialization failed.")
    
    final_answers: List[str] = []
    for question in request.questions:
        retrieved_clauses = searcher.search(question, k=5)
        answer = get_answer_from_llm(question, retrieved_clauses)
        final_answers.append(answer)
        
    return HackRxResponse(answers=final_answers)

@app.post("/api/v1/local/query", response_model=LocalQueryResponse)
async def local_query(request: LocalQueryRequest, _=Security(verify_token)):
    try:
        logging.info(f"Local query received for document: '{request.document_name}'")
        searcher = SemanticSearcher(request.document_name)
        if not searcher.index or not searcher.clauses:
            raise HTTPException(status_code=404, detail=f"Processed files for '{request.document_name}' not found. Run main.py first.")
        
        retrieved_clauses = searcher.search(request.question, k=5)
        answer = get_answer_from_llm(request.question, retrieved_clauses)
        
        return LocalQueryResponse(answer=answer, retrieved_clauses=retrieved_clauses)
    except Exception as e:
        logging.error(f"An error occurred in local_query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)