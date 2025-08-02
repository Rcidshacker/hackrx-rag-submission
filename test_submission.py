# test_submission.py (Corrected)

import requests
import json
from typing import Dict, List, Any

# --- Configuration ---
DOCUMENT_URL = "http://localhost:8080/Arogya_Sanjeevani_Policy.pdf"
QUESTIONS = [
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "How does the policy define a 'Hospital'?"
]
API_TOKEN = "a0f73b66d6d32b7707a37b571356dd469eadc2f5091c18ac75dd77ceee634a4c"
API_URL = "http://127.0.0.1:8000/api/v1/hackrx/run"

# --- Script ---
def run_submission_test():
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload: Dict[str, Any] = {
        "documents": DOCUMENT_URL,
        "questions": QUESTIONS
    }
    
    print("--- Simulating HackRx Submission ---")
    print(f"Sending document URL: {DOCUMENT_URL}")
    print(f"Sending {len(QUESTIONS)} questions...")
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        
        print("\n--- SUCCESS ---")
        print("Received response from the server:\n")
        response_data = response.json()
        print(json.dumps(response_data, indent=2))
        
    except requests.exceptions.RequestException as e:
        print("\n--- ERROR ---")
        print(f"An error occurred while making the request: {e}")
        if e.response:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response Body: {e.response.text}")

if __name__ == "__main__":
    run_submission_test()