"""Free-tier LLM factory. Set LLM_PROVIDER=groq (GROQ_API_KEY) or gemini (GOOGLE_API_KEY) in .env."""
from __future__ import annotations

import os

from thermalguard import config  # noqa: F401  (loads .env)

DEFAULT_MODELS = {"groq": "openai/gpt-oss-120b", "gemini": "gemini-2.5-flash"}
KEY_VARS = {"groq": "GROQ_API_KEY", "gemini": "GOOGLE_API_KEY"}


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").lower()


def llm_available() -> bool:
    return bool(os.getenv(KEY_VARS.get(provider(), ""), ""))


def get_llm(temperature: float = 0.0, max_tokens: int | None = None, model: str | None = None,
            reasoning_effort: str | None = None):
    p = provider()
    model = model or os.getenv("LLM_MODEL", DEFAULT_MODELS[p])
    if p == "groq":
        from langchain_groq import ChatGroq

        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        return ChatGroq(model=model, temperature=temperature, max_retries=3, max_tokens=max_tokens, **extra)
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=3,
                                      max_output_tokens=max_tokens)
    raise ValueError(f"unknown LLM_PROVIDER {p}")
