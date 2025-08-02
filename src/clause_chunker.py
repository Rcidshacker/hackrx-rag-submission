# src/clause_chunker.py (Corrected)

import re
import logging
from typing import List, Dict
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logging.info("Downloading 'punkt' tokenizer for NLTK...")
    nltk.download('punkt', quiet=True)

# ... (The rest of the file is the same and is functionally correct) ...
def is_new_chunk_header(line: str) -> bool:
    stripped_line = line.strip()
    if not stripped_line: return False
    strong_header_pattern = re.compile(r'^\s*(\d+(\.\d+)*\.\s+|[a-zA-Z]\)[\s\.]+|\(?[ivxlcdm]+\)[\s\.]+)')
    if strong_header_pattern.match(stripped_line): return True
    if stripped_line.isupper() and (1 <= len(stripped_line.split()) <= 7) and not stripped_line.endswith(('.', ':')): return True
    return False

def chunk_text_into_clauses(text: str, source_filename: str) -> List[Dict[str, str]]:
    initial_chunks: List[Dict[str, str]] = []
    lines = text.split('\n')
    current_chunk_text: List[str] = []
    for line in lines:
        if is_new_chunk_header(line) and current_chunk_text:
            header = current_chunk_text[0].strip()
            initial_chunks.append({"id": header, "text": "\n".join(current_chunk_text)})
            current_chunk_text = [line]
        else:
            current_chunk_text.append(line)
    if current_chunk_text:
        header = current_chunk_text[0].strip()
        initial_chunks.append({"id": header, "text": "\n".join(current_chunk_text)})
    final_clauses: List[Dict[str, str]] = []
    for chunk in initial_chunks:
        if len(chunk['text']) < 350:
            final_clauses.append({"clause_id": chunk['id'], "text": chunk['text'].strip().replace('\n', ' '), "source": source_filename})
        else:
            sentences: List[str] = nltk.sent_tokenize(chunk['text'])
            for i, sentence in enumerate(sentences):
                cleaned_sentence = sentence.replace('\n', ' ').strip()
                if len(cleaned_sentence) > 15:
                    final_clauses.append({"clause_id": f"{chunk['id']} (Sentence {i+1})", "text": cleaned_sentence, "source": source_filename})
    logging.info(f"  - Chunked document into {len(final_clauses)} granular clauses.")
    return final_clauses