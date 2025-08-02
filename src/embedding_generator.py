# src/embedding_generator.py

import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy.typing as npt

MODEL_NAME = 'all-mpnet-base-v2'

def generate_and_save_embeddings(clauses_data: List[Dict[str, str]], index_path: str) -> None:
    """
    Generates embeddings for text clauses and saves them to a FAISS index.
    """
    if not clauses_data:
        logging.warning("  - No clauses found to generate embeddings. Skipping.")
        return

    try:
        logging.info(f"  - Loading sentence transformer model: '{MODEL_NAME}'")
        model = SentenceTransformer(MODEL_NAME)
        
        texts = [clause['text'] for clause in clauses_data]
        
        logging.info(f"  - Generating embeddings for {len(texts)} clauses...")
        embeddings: npt.NDArray[np.float32] = model.encode(texts, show_progress_bar=True)
        
        # Ensure embeddings are float32, which FAISS expects.
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype('float32')
        
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        
        # Pylance may show a false positive here due to missing faiss stubs. The code is correct.
        index.add(embeddings)
        
        logging.info(f"  - Saving FAISS index to: {index_path}")
        faiss.write_index(index, index_path)
        
    except Exception as e:
        logging.error(f"  - An error occurred during embedding generation: {e}")