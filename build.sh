#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Download the NLTK 'punkt' tokenizer
python -c "import nltk; nltk.download('punkt')"