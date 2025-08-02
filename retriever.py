# retriever.py (Corrected)

import json
import logging
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy.typing as npt

# --- Configuration ---
MODEL_NAME = 'all-mpnet-base-v2'
CLAUSES_DIR = 'output_clauses'
EMBEDDINGS_DIR = 'output_embeddings'

logger = logging.getLogger(__name__)

class SemanticSearcher:
    def __init__(self, document_base_name: str) -> None:
        self.document_base_name = document_base_name
        self.model: SentenceTransformer
        self.clauses: List[Dict[str, str]] = []
        self.index: Optional[faiss.Index] = None

        clauses_path = os.path.join(CLAUSES_DIR, f"{document_base_name}_clauses.json")
        index_path = os.path.join(EMBEDDINGS_DIR, f"{document_base_name}.index")

        if not all(os.path.exists(p) for p in [clauses_path, index_path]):
            logger.error(f"Missing processed files for document '{document_base_name}'.")
            return

        logger.info(f"Loading model '{MODEL_NAME}' for semantic search...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.clauses = self._load_clauses(clauses_path)
        self.index = self._load_index(index_path)

    def _load_clauses(self, path: str) -> List[Dict[str, str]]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading clauses from {path}: {e}")
            return []

    def _load_index(self, path: str) -> Optional[faiss.Index]:
        try:
            # Pylance will warn about missing stubs, which is safe to ignore.
            return faiss.read_index(path)
        except Exception as e:
            logger.error(f"Error loading FAISS index from {path}: {e}")
            return None

    def search(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        if not self.clauses or self.index is None:
            logger.error("Searcher is not properly initialized.")
            return []

        query_embedding: npt.NDArray[np.float32] = self.model.encode([query])
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype('float32')

        # We only need the indices, so we can ignore the distances variable.
        _distances, indices = self.index.search(query_embedding, k)
        
        results: List[Dict[str, str]] = [self.clauses[idx] for idx in indices[0] if 0 <= idx < len(self.clauses)]
        return results

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    TEST_DOC = "Arogya_Sanjeevani_Policy" 
    TEST_QUERY = "What is the waiting period for pre-existing diseases?"
    logger.info(f"--- Running Standalone Retriever Test for document: {TEST_DOC} ---")
    searcher = SemanticSearcher(TEST_DOC)
    if searcher.index and searcher.clauses:
        retrieved_clauses = searcher.search(TEST_QUERY, k=3)
        print(json.dumps(retrieved_clauses, indent=2))