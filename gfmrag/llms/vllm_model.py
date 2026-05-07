import logging
import time

import requests
from openai import OpenAI

from .base_language_model import BaseLanguageModel

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class VLLMModel(BaseLanguageModel):
    """A class that interacts with a vLLM server through its OpenAI-compatible API.

    This class provides functionality to generate text using models served by vLLM
    while handling token limits, retries, and health checks.

    Args:
        model_name_or_path (str): The model name as registered in the vLLM server
            (e.g., "meta-llama/Meta-Llama-3-8B").
        base_url (str): The base URL of the vLLM server (e.g., "http://localhost:8000/v1").
        api_key (str): API key for the vLLM server. Defaults to "EMPTY".
        retry (int): Number of retries for failed API calls. Defaults to 5.
        timeout (int): Timeout in seconds for API calls. Defaults to 60.
        maximun_token (int | None): Maximum token limit for the model. If None,
            auto-detected from the server via GET /v1/models. Defaults to None.
        temperature (float): Temperature for text generation. Defaults to 0.0.

    Raises:
        RuntimeError: If the vLLM server is unreachable during initialization.
    """

    def __init__(
        self,
        model_name_or_path: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        retry: int = 5,
        timeout: int = 60,
        maximun_token: int | None = None,
        temperature: float = 0.0,
    ):
        self.model_name = model_name_or_path
        self.base_url = base_url.rstrip("/")
        self.retry = retry
        self.timeout = timeout
        self.temperature = temperature

        # Health check: verify the vLLM server is reachable
        if not self._is_api_available():
            raise RuntimeError(f"vLLM API is not available at {self.base_url}")

        # Auto-detect maximun_token from server if not provided
        if maximun_token is not None:
            self.maximun_token = maximun_token
        else:
            self.maximun_token = self._detect_max_model_len()

        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
        )

    def _is_api_available(self) -> bool:
        """Check if the vLLM server is reachable via GET /health."""
        try:
            health_url = self.base_url.replace("/v1", "") + "/health"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _detect_max_model_len(self) -> int:
        """Auto-detect max_model_len from GET /v1/models endpoint.

        Falls back to 4096 if the field is not present or the request fails.
        """
        try:
            models_url = f"{self.base_url}/models"
            response = requests.get(models_url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # The /v1/models endpoint returns {"data": [{"id": "...", ...}]}
            models = data.get("data", [])
            if models:
                model_info = models[0]
                max_len = model_info.get("max_model_len")
                if max_len is not None:
                    return int(max_len)
        except Exception as e:
            logger.warning(f"Failed to auto-detect max_model_len: {e}")

        return 4096

    def token_len(self, text: str) -> int:
        """Return the token count for the given text using the vLLM /tokenize endpoint.

        Falls back to AutoTokenizer.from_pretrained() if the /tokenize endpoint
        returns a 404 (not available in older vLLM versions).
        """
        try:
            tokenize_url = self.base_url.replace("/v1", "") + "/tokenize"
            response = requests.post(
                tokenize_url,
                json={"model": self.model_name, "prompt": text},
                timeout=10,
            )
            if response.status_code == 404:
                # Fallback to AutoTokenizer if /tokenize not available
                return self._token_len_fallback(text)
            response.raise_for_status()
            result = response.json()
            return result.get("count", len(result.get("tokens", [])))
        except requests.exceptions.HTTPError:
            return self._token_len_fallback(text)

    def _token_len_fallback(self, text: str) -> int:
        """Fallback tokenization using AutoTokenizer when /tokenize is unavailable."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return len(tokenizer.encode(text))

    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        """Generate a response using the vLLM server via the OpenAI-compatible API.

        This method sends a request to the vLLM server and returns the generated
        response. It handles both single string inputs and message lists, with
        retry logic for failed attempts.

        Args:
            llm_input (str | list): Either a string containing the user's input or
                a list of message dictionaries.
            system_input (str): System message to be prepended to the conversation.
                Defaults to "".

        Returns:
            str | Exception: The generated response text if successful, or the
                Exception if all retries fail.
        """
        # If the input is a list, it is assumed that the input is a list of messages
        if isinstance(llm_input, list):
            message = llm_input
        else:
            message = []
            if system_input:
                message.append({"role": "system", "content": system_input})
            message.append({"role": "user", "content": llm_input})

        # Check if the input is too long and truncate if necessary
        message_string = "\n".join([m["content"] for m in message])
        input_length = self.token_len(message_string)
        if input_length > self.maximun_token:
            logger.warning(
                f"Input length {input_length} exceeds max tokens "
                f"{self.maximun_token}. Truncating input."
            )
            llm_input = llm_input[: self.maximun_token]
            # Rebuild message after truncation
            message = []
            if system_input:
                message.append({"role": "system", "content": system_input})
            message.append({"role": "user", "content": llm_input})

        cur_retry = 0
        num_retry = self.retry
        error = Exception("Failed to generate sentence")
        while cur_retry <= num_retry:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    timeout=self.timeout,
                    temperature=self.temperature,
                )
                result = response.choices[0].message.content.strip()  # type: ignore
                return result
            except Exception as e:
                logger.error(
                    f"vLLM generation error (attempt {cur_retry + 1}/{num_retry + 1}): {e}"
                )
                time.sleep(30)
                cur_retry += 1
                error = e
                continue
        return error
