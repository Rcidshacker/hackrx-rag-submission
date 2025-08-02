# main.py (Upgraded with Caching Logic)

import os
import logging
import json
from datetime import datetime
from typing import List, Dict

# Import functions from our source modules
from src.file_handler import extract_text_from_pdf, extract_text_from_docx
from src.text_cleaner import load_cleaning_patterns, clean_text_with_patterns, post_process_text
from src.clause_chunker import chunk_text_into_clauses
from src.embedding_generator import generate_and_save_embeddings

# --- Configuration ---
INPUT_DIR = 'input_docs'
CLEANED_DIR = 'output_cleaned'
CLAUSES_DIR = 'output_clauses'
EMBEDDINGS_DIR = 'output_embeddings'
LOG_DIR = 'logs'
CONFIG_PATH = 'config/cleaning_patterns.yaml'

def setup_logging() -> None:
    """Configures logging to file and console."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(LOG_DIR, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def should_reprocess(source_path: str, clause_path: str, index_path: str) -> bool:
    """
    Checks if a document needs to be re-processed based on file modification times.
    """
    if not os.path.exists(clause_path) or not os.path.exists(index_path):
        logging.info("  - Output file(s) missing. Needs processing.")
        return True
    
    try:
        source_mod_time = os.path.getmtime(source_path)
        index_mod_time = os.path.getmtime(index_path)
        
        if source_mod_time > index_mod_time:
            logging.info("  - Source file is newer than the index. Needs reprocessing.")
            return True
    except OSError as e:
        logging.warning(f"  - Could not check file modification times: {e}. Reprocessing just in case.")
        return True
    
    return False

def main() -> None:
    """Main function to orchestrate the document processing workflow."""
    setup_logging()
    
    # Ensure necessary directories exist
    for directory in [INPUT_DIR, CLEANED_DIR, CLAUSES_DIR, EMBEDDINGS_DIR]:
        os.makedirs(directory, exist_ok=True)

    logging.info("--- Document Processing Workflow Started ---")
    
    cleaning_patterns = load_cleaning_patterns(CONFIG_PATH)
    if not cleaning_patterns:
        logging.error("Could not load cleaning patterns. Aborting.")
        return

    try:
        files_to_process = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
    except FileNotFoundError:
        logging.error(f"Input directory '{INPUT_DIR}' not found. Please create it and add documents.")
        return
    
    if not files_to_process:
        logging.warning(f"No files found in '{INPUT_DIR}'. Add documents to process.")
        return

    processed_count, skipped_count, error_count = 0, 0, 0

    for filename in files_to_process:
        file_path = os.path.join(INPUT_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        
        clauses_output_path = os.path.join(CLAUSES_DIR, f"{base_name}_clauses.json")
        index_output_path = os.path.join(EMBEDDINGS_DIR, f"{base_name}.index")

        logging.info(f"\nProcessing file: {filename}")

        if not should_reprocess(file_path, clauses_output_path, index_output_path):
            logging.info("  - Output files are up-to-date. Skipping.")
            skipped_count += 1
            continue

        raw_text: str | None = None
        if filename.lower().endswith('.pdf'):
            raw_text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith('.docx'):
            raw_text = extract_text_from_docx(file_path)
        else:
            logging.warning(f"  - Skipping unsupported file type: {filename}")
            skipped_count += 1
            continue

        if raw_text:
            # STAGE 1: CLEANING
            text_after_patterns = clean_text_with_patterns(raw_text, cleaning_patterns)
            final_cleaned_text = post_process_text(text_after_patterns)
            
            cleaned_output_path = os.path.join(CLEANED_DIR, f"{base_name}_cleaned.txt")
            with open(cleaned_output_path, 'w', encoding='utf-8') as f:
                f.write(final_cleaned_text)
            logging.info(f"  - Intermediate cleaned text saved to: {cleaned_output_path}")

            # STAGE 2: CHUNKING
            clauses: List[Dict[str, str]] = chunk_text_into_clauses(final_cleaned_text, filename)
            
            with open(clauses_output_path, 'w', encoding='utf-8') as f:
                json.dump(clauses, f, indent=4)
            logging.info(f"  - Successfully chunked and saved to: {clauses_output_path}")

            # STAGE 3: EMBEDDING
            generate_and_save_embeddings(clauses, index_output_path)
            processed_count += 1
        else:
            logging.error(f"  - Text extraction failed for {filename}.")
            error_count += 1

    logging.info("\n--- Processing Summary ---")
    logging.info(f"Total files processed/updated: {processed_count}")
    logging.info(f"Total files skipped (up-to-date or unsupported): {skipped_count}")
    logging.info(f"Total files with errors: {error_count}")
    logging.info("--------------------------")

if __name__ == "__main__":
    main()