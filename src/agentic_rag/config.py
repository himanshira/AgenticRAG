"""Central place for environment, paths and the LLM instance."""

import os
from pathlib import Path

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = Path(os.getenv("PDF_PATH", PROJECT_ROOT / "data" / "domain_docs.pdf"))
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

# Describe what the static PDF covers. The router reads this to decide.
DOMAIN_DESCRIPTION = (
    "the paper 'Attention Is All You Need' (Vaswani et al., 2017): the Transformer "
    "architecture, self-attention and scaled dot-product attention, multi-head attention, "
    "positional encoding, the encoder-decoder structure, training setup, and its "
    "machine-translation results"
)


def get_llm() -> LLM:
    return LLM(model=LLM_MODEL, temperature=0.2)
