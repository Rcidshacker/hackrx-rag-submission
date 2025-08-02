# src/text_cleaner.py

import yaml
import re
import logging
from typing import List, Dict, Any, Pattern

def load_cleaning_patterns(yaml_path: str = 'config/cleaning_patterns.yaml') -> List[Pattern[str]]:
    """
    Loads and compiles regex patterns from a YAML configuration file.
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            patterns_config: Any = yaml.safe_load(f)
        
        compiled_patterns: List[Pattern[str]] = []
        if not isinstance(patterns_config, dict):
            logging.error(f"YAML file '{yaml_path}' is not structured as a dictionary.")
            return []

        for category, patterns in patterns_config.items():
            if not isinstance(patterns, list):
                logging.warning(f"Category '{category}' in YAML is not a list. Skipping.")
                continue
            for pattern_str in patterns:
                if isinstance(pattern_str, str):
                    try:
                        compiled_patterns.append(re.compile(pattern_str, re.IGNORECASE))
                    except re.error as e:
                        logging.error(f"Invalid regex in category '{category}': '{pattern_str}'. Error: {e}")
        
        logging.info(f"Successfully loaded and compiled {len(compiled_patterns)} cleaning patterns.")
        return compiled_patterns
    except FileNotFoundError:
        logging.error(f"Cleaning patterns file not found at '{yaml_path}'.")
        return []
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file '{yaml_path}': {e}")
        return []

def clean_text_with_patterns(text: str, compiled_patterns: List[Pattern[str]]) -> str:
    """
    Removes entire lines from text that match any of the compiled regex patterns.
    """
    if not text or not compiled_patterns:
        return text

    cleaned_lines: List[str] = []
    for line in text.split('\n'):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        is_noise = any(pattern.fullmatch(stripped_line) for pattern in compiled_patterns)
        
        if not is_noise:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def post_process_text(text: str) -> str:
    """
    Applies final cleaning steps like fixing hyphenation and normalizing whitespace.
    """
    processed_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    processed_text = re.sub(r'[ \t]+', ' ', processed_text)
    processed_text = re.sub(r'\n{3,}', '\n\n', processed_text)
    return processed_text.strip()