"""Comprehensive unit tests for the Gemini LLM client wrapper and retry logic.

Verifies:
- API key retrieval from environment variables and Streamlit secrets.
- Live Gemini availability detection against empty and placeholder keys.
- Client instantiation and graceful fallback to None when unconfigured.
- Detection of transient vs non-transient API and network errors.
- Retry loop behavior on first-attempt success, transient recovery, and retry exhaustion.
- Immediate failure without retry on non-transient errors.
- Argument pass-through to client.models.generate_content.
"""

import os
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import requests
from google import genai
from google.genai import errors

from src.llm import (
    DEFAULT_MODEL,
    generate_content_with_retry,
    get_api_key,
    get_gemini_client,
    is_live_gemini_available,
    is_transient_error,
)


class TestApiKeyAndAvailability:
    """Test suite for get_api_key() and is_live_gemini_available()."""

    def test_get_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch):
        """Valid GEMINI_API_KEY from environment is retrieved successfully."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-live-key-xyz123")
        assert get_api_key() == "test-live-key-xyz123"

    def test_get_api_key_strips_whitespace(self, monkeypatch: pytest.MonkeyPatch):
        """Leading and trailing whitespace in GEMINI_API_KEY is trimmed."""
        monkeypatch.setenv("GEMINI_API_KEY", "   key-with-spaces   \n")
        assert get_api_key() == "key-with-spaces"

    def test_get_api_key_missing_or_empty(self, monkeypatch: pytest.MonkeyPatch):
        """Empty string or absent environment variable returns None."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        # Ensure streamlit fallback does not trigger
        with patch.dict("sys.modules", {"streamlit": None}):
            assert get_api_key() is None

        monkeypatch.setenv("GEMINI_API_KEY", "   ")
        with patch.dict("sys.modules", {"streamlit": None}):
            assert get_api_key() is None

    def test_get_api_key_streamlit_secrets_fallback(self, monkeypatch: pytest.MonkeyPatch):
        """Fallback to streamlit.secrets when environment variable is absent."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        mock_st = ModuleType("streamlit")
        mock_st.secrets = {"GEMINI_API_KEY": "streamlit-secret-key-456"}  # type: ignore

        with patch.dict("sys.modules", {"streamlit": mock_st}):
            assert get_api_key() == "streamlit-secret-key-456"

    @pytest.mark.parametrize(
        "placeholder",
        [
            "your_gemini_api_key_here",
            "YOUR_GEMINI_API_KEY_HERE",
            "placeholder",
            "none",
            "null",
            "",
            "   ",
        ],
    )
    def test_is_live_gemini_available_with_placeholders(
        self, placeholder: str, monkeypatch: pytest.MonkeyPatch
    ):
        """Placeholder and blank values must result in is_live_gemini_available() returning False."""
        monkeypatch.setenv("GEMINI_API_KEY", placeholder)
        with patch.dict("sys.modules", {"streamlit": None}):
            assert is_live_gemini_available() is False

    def test_is_live_gemini_available_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        """When GEMINI_API_KEY is unset and no secrets exist, returns False."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch.dict("sys.modules", {"streamlit": None}):
            assert is_live_gemini_available() is False

    def test_is_live_gemini_available_with_real_key(self, monkeypatch: pytest.MonkeyPatch):
        """A valid non-placeholder key results in is_live_gemini_available() returning True."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyB3ValidLookingProductionKey")
        assert is_live_gemini_available() is True


class TestGetGeminiClient:
    """Test suite for get_gemini_client() factory."""

    def test_get_gemini_client_none_when_unavailable(self, monkeypatch: pytest.MonkeyPatch):
        """Returns None when credentials are not configured or are placeholder."""
        monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
        assert get_gemini_client() is None

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch.dict("sys.modules", {"streamlit": None}):
            assert get_gemini_client() is None

    def test_get_gemini_client_success(self, monkeypatch: pytest.MonkeyPatch):
        """Instantiates and returns genai.Client when valid credentials exist."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestClientKey123")
        client = get_gemini_client()
        assert client is not None
        assert isinstance(client, genai.Client)

    def test_get_gemini_client_handles_exception(self, monkeypatch: pytest.MonkeyPatch):
        """Returns None gracefully if genai.Client constructor raises an exception."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyValidFormatKey")
        with patch("google.genai.Client", side_effect=RuntimeError("SDK init failure")):
            client = get_gemini_client()
            assert client is None


class TestIsTransientError:
    """Test suite for transient vs non-transient error classification."""

    @pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
    def test_transient_status_codes(self, status_code: int):
        """Exceptions with transient HTTP status codes are flagged as transient."""
        exc = Exception(f"HTTP error with code {status_code}")
        setattr(exc, "status_code", status_code)
        assert is_transient_error(exc) is True

        exc_code = Exception(f"APIError with code {status_code}")
        setattr(exc_code, "code", status_code)
        assert is_transient_error(exc_code) is True

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
    def test_non_transient_client_status_codes(self, status_code: int):
        """Standard 4xx client errors (excluding 408/429) are non-transient."""
        exc = Exception(f"Client error {status_code}")
        setattr(exc, "status_code", status_code)
        assert is_transient_error(exc) is False

    def test_google_genai_server_error_is_transient(self):
        """errors.ServerError subclass is transient."""
        err = errors.ServerError(503, {"message": "Service unavailable"})
        assert is_transient_error(err) is True

    def test_google_genai_client_error_429_is_transient(self):
        """errors.ClientError with code 429 is transient."""
        err = errors.ClientError(429, {"message": "Rate limit exceeded"})
        assert is_transient_error(err) is True

    def test_google_genai_client_error_400_is_not_transient(self):
        """errors.ClientError with code 400 is not transient."""
        err = errors.ClientError(400, {"message": "Bad request invalid argument"})
        assert is_transient_error(err) is False

    def test_network_and_transport_exceptions(self):
        """Standard network, timeout, and connection errors are transient."""
        assert is_transient_error(ConnectionError("Connection aborted")) is True
        assert is_transient_error(TimeoutError("Request timed out")) is True
        assert is_transient_error(requests.exceptions.ConnectionError("Connection refused")) is True
        assert is_transient_error(requests.exceptions.Timeout("Read timed out")) is True

    def test_message_token_inspection(self):
        """Exceptions with transient keywords in message are detected."""
        assert is_transient_error(RuntimeError("429 Resource Exhausted")) is True
        assert is_transient_error(RuntimeError("Model overloaded. Please retry later.")) is True
        assert is_transient_error(RuntimeError("Service Unavailable 503")) is True

    def test_non_transient_exceptions(self):
        """Validation errors and general non-network errors are non-transient."""
        assert is_transient_error(ValueError("Invalid argument")) is False
        assert is_transient_error(TypeError("Expected str, got int")) is False
        assert is_transient_error(None) is False  # type: ignore


class TestGenerateContentWithRetry:
    """Test suite for generate_content_with_retry() execution and backoff logic."""

    def test_rejects_none_client(self):
        """Raises ValueError when client is None."""
        with pytest.raises(ValueError, match="client parameter cannot be None"):
            generate_content_with_retry(client=None)

    def test_success_on_first_attempt(self):
        """Returns result immediately when the first call succeeds."""
        mock_client = MagicMock()
        expected_response = MagicMock()
        mock_client.models.generate_content.return_value = expected_response

        with patch("time.sleep") as mock_sleep:
            result = generate_content_with_retry(
                client=mock_client,
                model="gemini-2.5-flash",
                contents="Analyze notice",
            )
            assert result == expected_response
            assert mock_client.models.generate_content.call_count == 1
            mock_sleep.assert_not_called()

    def test_retry_on_transient_error_and_subsequent_success(self):
        """Recovers from a transient error on attempt 1 and succeeds on attempt 2."""
        mock_client = MagicMock()
        expected_response = MagicMock()

        transient_err = errors.ServerError(503, {"message": "Backend temporarily unavailable"})
        mock_client.models.generate_content.side_effect = [
            transient_err,
            expected_response,
        ]

        with patch("time.sleep") as mock_sleep:
            result = generate_content_with_retry(
                client=mock_client,
                model="gemini-2.5-flash",
                contents="Analyze notice",
                max_retries=2,
                backoff_factor=1.0,
            )
            assert result == expected_response
            assert mock_client.models.generate_content.call_count == 2
            # Attempt 0 failed -> backoff_factor * (2 ** 0) = 1.0s
            mock_sleep.assert_called_once_with(1.0)

    def test_exhausts_retries_and_raises(self):
        """Raises original exception after exhausting max_retries on persistent transient error."""
        mock_client = MagicMock()
        transient_err = errors.ClientError(429, {"message": "Quota exceeded"})
        mock_client.models.generate_content.side_effect = transient_err

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(errors.ClientError) as exc_info:
                generate_content_with_retry(
                    client=mock_client,
                    max_retries=2,
                    backoff_factor=1.5,
                )

            assert exc_info.value == transient_err
            # Total attempts = 1 initial + 2 retries = 3
            assert mock_client.models.generate_content.call_count == 3
            # Two sleep calls: 1.5 * (2**0) = 1.5s, 1.5 * (2**1) = 3.0s
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1.5)
            mock_sleep.assert_any_call(3.0)

    def test_immediate_failure_on_non_transient_error(self):
        """Non-transient errors (e.g. ValueError, 400 Bad Request) fail immediately without retry."""
        mock_client = MagicMock()
        non_transient_err = ValueError("Invalid prompt format")
        mock_client.models.generate_content.side_effect = non_transient_err

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError) as exc_info:
                generate_content_with_retry(
                    client=mock_client,
                    max_retries=3,
                )

            assert exc_info.value == non_transient_err
            assert mock_client.models.generate_content.call_count == 1
            mock_sleep.assert_not_called()

    def test_passes_arguments_through_to_models_generate_content(self):
        """model, contents, config, and additional kwargs are properly passed to client."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock()

        dummy_config = {"temperature": 0.2}
        generate_content_with_retry(
            client=mock_client,
            model="gemini-2.5-pro",
            contents=["image_part", "text_prompt"],
            config=dummy_config,
            extra_param="test_val",
        )

        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-2.5-pro",
            contents=["image_part", "text_prompt"],
            config=dummy_config,
            extra_param="test_val",
        )
