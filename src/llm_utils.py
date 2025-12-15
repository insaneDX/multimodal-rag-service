"""
LLM inference utilities - separated to avoid circular imports.
"""
import os
from typing import Optional

import src.logger_config
from loguru import logger

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ---- Config ----
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")  # "hf", "openai", or "groq"
MODEL_NAME = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- LLM Client Initialization ----
_llm_client = None
_llm_pipe = None


def _init_groq_client():
    """Initialize Groq client."""
    global _llm_client
    if _llm_client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        from groq import Groq
        _llm_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized with model: {}", MODEL_NAME)
    return _llm_client


def _init_hf_pipeline():
    """Initialize HuggingFace pipeline."""
    global _llm_pipe
    if _llm_pipe is None:
        from transformers import pipeline
        _llm_pipe = pipeline("text2text-generation", model=MODEL_NAME)
        logger.info("HuggingFace pipeline initialized with model: {}", MODEL_NAME)
    return _llm_pipe


def _init_openai_client():
    """Initialize OpenAI client."""
    global _llm_client
    if _llm_client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        from openai import OpenAI
        _llm_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized with model: {}", MODEL_NAME)
    return _llm_client


def llm_generate(prompt: str, max_tokens: int = 512, temperature: float = 0.5) -> str:
    """
    Generate text using the configured LLM backend.
    
    Args:
        prompt: The input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text string
    """
    if LLM_BACKEND == "groq":
        return _generate_groq(prompt, max_tokens, temperature)
    elif LLM_BACKEND == "hf":
        return _generate_hf(prompt, max_tokens)
    elif LLM_BACKEND == "openai":
        return _generate_openai(prompt, max_tokens, temperature)
    else:
        raise ValueError(f"Unsupported LLM Backend: {LLM_BACKEND}")


def _generate_groq(prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate using Groq API."""
    try:
        client = _init_groq_client()
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL_NAME,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise


def _generate_hf(prompt: str, max_tokens: int) -> str:
    """Generate using HuggingFace pipeline."""
    try:
        pipe = _init_hf_pipeline()
        out = pipe(prompt, max_length=max_tokens, truncation=True)
        return out[0]["generated_text"]
    except Exception as e:
        logger.error(f"HuggingFace pipeline error: {e}")
        raise


def _generate_openai(prompt: str, max_tokens: int, temperature: float) -> str:
    """Generate using OpenAI API."""
    try:
        client = _init_openai_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise


def get_llm_backend() -> str:
    """Return the current LLM backend name."""
    return LLM_BACKEND


def get_model_name() -> str:
    """Return the current model name."""
    return MODEL_NAME