from __future__ import annotations

import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_ROOT / "data"
DEFAULT_CACHE_PATH = DATA_DIR / "cache.sqlite"

DEFAULT_LLM_MODEL = os.environ.get("NER_LLM_MODEL", "gpt-oss-120b")
DEFAULT_LLM_URL = os.environ.get(
    "NER_LLM_URL",
    "http://llm2.priv.bmi.emory.edu:8000/gpt-oss-120b/v1/completions",
)
DEFAULT_MAX_TOKENS = int(os.environ.get("NER_MAX_TOKENS", "1000"))
DEFAULT_BATCH_SIZE = int(os.environ.get("NER_BATCH_SIZE", "8"))
DEFAULT_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("NER_REQUEST_TIMEOUT_SECONDS", "120"))

DEFAULT_LABELS = [
    "Heart Failure = Yes",
    "Heart Failure = No",
    "Cyanosis",
    "FALD",
    "Dysrhythmia",
    "PLE or PB",
    "Phenotype date",
    "Fontan procedure date",
]

DEFAULT_PROMPT = """Extract named entities from the text. Your response MUST be valid JSON only, with no prose, markdown, or code fences.

Allowed labels:
{{labels}}

Output format:
{"entities":[{"label":"<one allowed label>","text":"<exact text from input>","context":"<short surrounding text from input>"}]}

Rules:
- Use only the allowed labels listed above.
- The "text" value must be copied exactly from the input.
- Return an empty entities array if no relevant entities are present.
- Do not include any explanation, commentary, or text outside the JSON object.

Input text:
{{text}}
"""

