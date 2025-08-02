# src/file_handler.py

import fitz  # PyMuPDF
from docx import Document
import logging
import os
from typing import Optional

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extracts raw text from a PDF file using PyMuPDF.
    """
    all_text = ""
    try:
        # Pylance may warn about 'fitz' having no type stubs, which is safe to ignore.
        with fitz.open(pdf_path) as doc:
            logging.info(f"  - Opened PDF: {os.path.basename(pdf_path)}, {doc.page_count} pages.")
            for page in doc: # Iterating through doc is the standard PyMuPDF practice
                try:
                    all_text += page.get_text("text") + "\n"
                except Exception as page_error:
                    logging.warning(f"  - Error on page {page.number + 1}: {page_error}")
        return all_text
    except Exception as e:
        logging.error(f"  - Failed to process PDF '{os.path.basename(pdf_path)}': {e}")
        return None

def extract_text_from_docx(docx_path: str) -> Optional[str]:
    """
    Extracts raw text from a DOCX file using python-docx.
    """
    try:
        document = Document(docx_path)
        logging.info(f"  - Opened DOCX: {os.path.basename(docx_path)}")
        all_text = [para.text for para in document.paragraphs if para.text]
        return "\n".join(all_text)
    except Exception as e:
        logging.error(f"  - Failed to process DOCX '{os.path.basename(docx_path)}': {e}")
        return None