# src/llm_handler.py (Final Version with NVIDIA Llama 3.1)

import os
import logging
import requests
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("OPENROUTER_SITE_URL")

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")
if not SITE_URL:
    raise ValueError("OPENROUTER_SITE_URL not found in .env file")

def get_answer_from_llm(query: str, retrieved_clauses: List[Dict[str, str]]) -> str:
    """
    Generates a concise answer using the NVIDIA Llama 3.1 model via OpenRouter.
    """
    if not retrieved_clauses:
        return "Based on the document, there is not enough information to answer this question."

    context_string = "\n\n".join(
        [f"Clause ID: {c.get('clause_id', 'N/A')}\nText: {c.get('text', '')}" for c in retrieved_clauses]
    )

    # This prompt is specifically designed for the NVIDIA Nemotron model.
    # It includes the required "detailed thinking on" instruction.
    system_prompt = (
        "You are an expert insurance policy analyst. Your task is to provide a direct and factual answer to the user's question based "
        "ONLY on the provided context clauses from an insurance policy. Do not use any external knowledge. "
        "If the context does not contain the necessary information, explicitly state that the information is not available in the provided document. "
        "Before providing the final answer, you must perform detailed thinking on the context and the question to ensure accuracy."
    )

    user_prompt = f"""
    **Context Clauses:**
    ---
    {context_string}
    ---

    **Question:**
    {query}
    """

    try:
        logging.info("Sending request to OpenRouter API with NVIDIA Llama 3.1 model...")
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "HTTP-Referer": SITE_URL,
                "X-Title": "HackRx RAG System"
            },
            json={
                "model": "openrouter/horizon-alpha", # <-- THE NEW, CORRECT MODEL NAME
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
            }
        )
        response.raise_for_status()
        response_json = response.json()
        answer = response_json['choices'][0]['message']['content']
        logging.info("Successfully received response from LLM.")
        return answer.strip() if answer else "No answer was generated."

    except requests.RequestException as e:
        logging.error(f"An error occurred while querying the OpenRouter API: {e}")
        return "Error: Could not connect to the language model service."
    except (KeyError, IndexError) as e:
        logging.error(f"Error parsing LLM response: {e}")
        return "Error: Invalid response format from the language model."