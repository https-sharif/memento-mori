from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"

MOCK_LLM = os.environ.get("MOCK_LLM", "").strip().lower() in ("1", "true", "yes")

if MOCK_LLM:
    logging.basicConfig(level=logging.INFO)
    logger.info("MOCK_LLM is set — using canned responses instead of calling Gemini.")
    from mock_llm import MockClient

    client = MockClient()
else:
    from google import genai

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY before running, or set MOCK_LLM=true to run without Gemini.")

    client = genai.Client()
