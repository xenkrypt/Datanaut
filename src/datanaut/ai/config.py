"""Every provider setting, limit and default for the AI analyst, in one place.

Model names are read from here only. Check them against each provider's current documentation
before a real run; the request formats are in `llm.py`.
"""

from __future__ import annotations

from collections.abc import Mapping

from datanaut import limits

# --- providers ----------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
PROVIDERS = ("gemini", "groq")
KEY_NAMES: Mapping[str, str] = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
"""Environment variable (or Streamlit secret) names that hold each provider's key."""

# --- request settings ---------------------------------------------------------------------
TIMEOUT_SECONDS = 30.0
MAX_OUTPUT_TOKENS = 700
TEMPERATURE = 0.0

# --- session limits -----------------------------------------------------------------------
SESSION_QUESTION_CAP = limits.AI_QUESTION_CAP
MIN_SECONDS_BETWEEN_CALLS = 2.0
MAX_REPAIR_RETRIES = 1
MAX_CALLS_PER_QUESTION = 6
"""query, one repair, explanation and one rewrite, with a little room."""
CACHE_SIZE = 64

# --- queries ------------------------------------------------------------------------------
DEFAULT_LIMIT = 50
MAX_LIMIT = 1000
MAX_IN_VALUES = 50
MAX_FILTERS = 8

# --- text going to the model --------------------------------------------------------------
MAX_COLUMN_NAME = 64
MAX_STRING = 100
MAX_FINDING_TEXT = 300
MAX_QUESTION = 500
MAX_CATEGORY_VALUES = 12
DEFAULT_SAMPLE_ROWS = 20
MAX_SAMPLE_ROWS = 50
MAX_LOG_ENTRIES = 25

# --- number check -------------------------------------------------------------------------
SIGNIFICANT_DIGITS = 2
