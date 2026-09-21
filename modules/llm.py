"""
modules/llm.py
==============
Phase 6 will implement this module.

Responsibilities:
-----------------
- Load the LLM provider and API key from environment variables.
- Build structured prompts that combine:
    · Clustered feedback samples
    · Retrieved RAG context from the knowledge base
    · Instructions for generating feature proposals
- Send requests to the configured LLM (e.g., Google Gemini API).
- Handle API errors, rate limits, and timeouts gracefully.
- Return the raw LLM response text for further parsing.

Planned dependencies (to be added in Phase 6):
    google-generativeai   (for Gemini)
    openai                (optional, if OpenAI support is added)
"""

import os
from typing import Optional


def get_llm_config() -> dict:
    """
    Read LLM configuration from environment variables.

    Returns:
        dict: Configuration containing provider name and API key.

    Raises:
        EnvironmentError: If required environment variables are not set.
    """
    # TODO (Phase 6): Read LLM_PROVIDER and GEMINI_API_KEY from os.environ.
    raise NotImplementedError("get_llm_config() will be implemented in Phase 6.")


def build_prompt(
    cluster_feedback: list,
    rag_context: str,
    theme_name: Optional[str] = None,
) -> str:
    """
    Construct the LLM prompt from clustered feedback and retrieved context.

    Args:
        cluster_feedback (list): List of feedback strings from a single cluster.
        rag_context (str): Relevant product knowledge retrieved via RAG.
        theme_name (Optional[str]): Human-readable cluster/theme label.

    Returns:
        str: The complete prompt string to send to the LLM.
    """
    # TODO (Phase 6): Format and return a structured prompt template.
    raise NotImplementedError("build_prompt() will be implemented in Phase 6.")


def call_llm(prompt: str, config: dict) -> str:
    """
    Send the prompt to the configured LLM and return the response text.

    Args:
        prompt (str): The structured input prompt.
        config (dict): LLM configuration (provider, API key, model settings).

    Returns:
        str: Raw text response from the LLM.
    """
    # TODO (Phase 6): Call the appropriate LLM API and return the response.
    raise NotImplementedError("call_llm() will be implemented in Phase 6.")
