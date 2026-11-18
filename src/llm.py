"""Gemini client wrapper with retry logic and credential management for NoticeWise.

Provides:
- API key detection with environment variables and Streamlit secrets fallback.
- Availability checking distinguishing between valid keys and placeholders.
- Google GenAI SDK client instantiation (genai.Client).
- Exponential backoff retry wrapper for multimodal content generation.
"""

import logging
import os
import time
from typing import Any, List, Optional, Set, Tuple, Type

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "gemini-2.5-flash"

# Dynamically gather transport / network exception types without hard-failing
_transient_exc_list: List[Type[Exception]] = [ConnectionError, TimeoutError, OSError]

try:
    import httpx  # type: ignore

    _transient_exc_list.extend([httpx.TransportError, httpx.TimeoutException])
except ImportError:
    pass

try:
    import httpx2  # type: ignore

    _transient_exc_list.extend([httpx2.TransportError, httpx2.TimeoutException])
except ImportError:
    pass

try:
    import requests  # type: ignore

    _transient_exc_list.append(requests.exceptions.RequestException)
except ImportError:
    pass

TRANSIENT_NETWORK_EXCEPTIONS: Tuple[Type[Exception], ...] = tuple(_transient_exc_list)

# Common template placeholders to treat as unconfigured/offline
PLACEHOLDER_KEYS: Set[str] = {
    "your_gemini_api_key_here",
    "placeholder",
    "none",
    "null",
    "",
}


def get_api_key() -> Optional[str]:
    """Retrieve Gemini API key from environment variable or Streamlit secrets.

    Checks:
    1. os.getenv("GEMINI_API_KEY") (after loading .env if present)
    2. streamlit.secrets["GEMINI_API_KEY"] (if streamlit is active and configured)

    Returns:
        Stripped string API key if present and non-empty, otherwise None.
    """
    load_dotenv()

    env_key = os.getenv("GEMINI_API_KEY")
    if env_key is not None:
        cleaned = env_key.strip()
        if cleaned:
            return cleaned

    # Fallback to Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st  # type: ignore

        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            secret_key = str(st.secrets["GEMINI_API_KEY"]).strip()
            if secret_key:
                return secret_key
    except Exception:
        # Streamlit not installed, not running, or secrets unconfigured
        pass

    return None


def is_live_gemini_available() -> bool:
    """Check if a valid, non-placeholder Gemini API key is configured.

    Returns:
        True if an active API key is present and not equal to template placeholders,
        False otherwise.
    """
    key = get_api_key()
    if not key:
        return False
    if key.lower() in PLACEHOLDER_KEYS:
        return False
    return True


def get_gemini_client() -> Optional[genai.Client]:
    """Initialize and return a Google GenAI Client instance if credentials are valid.

    Returns:
        genai.Client instance if key is available, None if offline/unconfigured.
    """
    if not is_live_gemini_available():
        logger.info(
            "Gemini API key is not available or is a placeholder; client returning None"
        )
        return None

    api_key = get_api_key()
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        logger.error("Failed to instantiate genai.Client: %s", exc)
        return None


def is_transient_error(exc: Exception) -> bool:
    """Determine whether an error is transient and eligible for retry.

    Matches HTTP 429 (rate limit / quota), 500, 502, 503, 504 (server errors),
    as well as network disconnects, timeouts, and connection resets.
    """
    if exc is None:
        return False

    # Check status code or code attribute
    code = getattr(exc, "code", None)
    if code is None:
        code = getattr(exc, "status_code", None)
    if code in (408, 429, 500, 502, 503, 504):
        return True

    # Check google.genai.errors classes
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError) and getattr(exc, "code", None) == 429:
        return True

    # Network and transport errors
    if isinstance(exc, TRANSIENT_NETWORK_EXCEPTIONS):
        return True

    # String inspection fallback for wrapped or custom exceptions
    msg = str(exc).lower()
    transient_tokens = (
        "429",
        "rate limit",
        "resource_exhausted",
        "resource exhausted",
        "quota",
        "too many requests",
        "503",
        "service unavailable",
        "unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "connection error",
        "temporarily unavailable",
        "overloaded",
    )
    return any(token in msg for token in transient_tokens)


def generate_content_with_retry(
    client: Any,
    model: str = DEFAULT_MODEL,
    contents: Any = None,
    config: Any = None,
    max_retries: int = 2,
    backoff_factor: float = 1.5,
    **kwargs: Any,
) -> Any:
    """Invoke client.models.generate_content with exponential backoff retry on transient errors.

    Args:
        client: genai.Client instance (must provide .models.generate_content).
        model: Model name string, defaults to 'gemini-2.5-flash'.
        contents: Input prompt or multimodal payload (text, images, parts, etc.).
        config: Optional types.GenerateContentConfig or configuration dict.
        max_retries: Maximum number of retry attempts after initial failure (default 2).
        backoff_factor: Base multiplier for exponential backoff (default 1.5s).
        **kwargs: Additional parameters passed through to generate_content.

    Returns:
        types.GenerateContentResponse from Gemini API.

    Raises:
        ValueError: If client is None.
        Exception: If retries are exhausted or a non-transient exception occurs.
    """
    if client is None:
        raise ValueError("client parameter cannot be None")

    total_attempts = max_retries + 1
    for attempt in range(total_attempts):
        try:
            logger.debug(
                "Calling generate_content (attempt %d/%d, model=%s)",
                attempt + 1,
                total_attempts,
                model,
            )
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
                **kwargs,
            )
        except Exception as exc:
            is_transient = is_transient_error(exc)
            if attempt < max_retries and is_transient:
                sleep_delay = backoff_factor * (2**attempt)
                logger.warning(
                    "Transient error on attempt %d/%d: %s. Retrying in %.2fs...",
                    attempt + 1,
                    total_attempts,
                    exc,
                    sleep_delay,
                )
                time.sleep(sleep_delay)
            else:
                logger.error(
                    "Generate content failed on attempt %d/%d (transient=%s, max_retries=%d): %s",
                    attempt + 1,
                    total_attempts,
                    is_transient,
                    max_retries,
                    exc,
                )
                raise
