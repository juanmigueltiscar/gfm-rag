# Adapt from: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/named_entity_extraction_parallel.py
import logging
from typing import Literal

from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from gfmrag.graph_index_construction.langchain_util import init_langchain_model
from gfmrag.graph_index_construction.utils import extract_json_dict, processing_phrases

from .base_model import BaseNERModel

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

query_prompt_one_shot_input = """Por favor, extrae todas las entidades nombradas que son importantes para responder las preguntas siguientes.
Coloca las entidades nombradas en formato json.

Pregunta: ¿Qué empresa adoptó antes la impresión inkjet para decoración cerámica, Keraben o Porcelanosa?

"""
query_prompt_one_shot_output = """
{"named_entities": ["impresión inkjet", "decoración cerámica", "Keraben", "Porcelanosa"]}
"""

query_prompt_template = """
Pregunta: {}

"""


class LLMNERModel(BaseNERModel):
    """A Named Entity Recognition (NER) model that uses Language Models (LLMs) for entity extraction.

    This class implements entity extraction using various LLM backends (OpenAI, Together, Ollama, llama.cpp, vLLM)
    through the Langchain interface. It processes text input and returns a list of extracted named entities.

    Args:
        llm_api (Literal["openai", "nvidia", "together", "ollama", "llama.cpp", "vllm"]): The LLM backend to use. Defaults to "openai".
        model_name (str): Name of the specific model to use. Defaults to "gpt-4o-mini".
        max_tokens (int): Maximum number of tokens in the response. Defaults to 1024.
        json_mode (bool): Whether to use JSON mode (response_format) for structured output. Defaults to True.
        base_url (str | None): Base URL for vLLM server. Used when llm_api="vllm". Defaults to None.
        api_key (str | None): API key for vLLM server. Used when llm_api="vllm". Defaults to None.

    Methods:
        __call__: Extracts named entities from the input text.

    Raises:
        Exception: If there's an error in extracting or processing named entities.
    """

    def __init__(
        self,
        llm_api: Literal[
            "openai", "nvidia", "together", "ollama", "llama.cpp", "vllm"
        ] = "openai",
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        json_mode: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
        num_runs: int = 1,
    ):
        """Initialize the LLM-based NER model.

        Args:
            llm_api (Literal["openai", "nvidia", "together", "ollama", "llama.cpp", "vllm"]): The LLM API provider to use.
                Defaults to "openai".
            model_name (str): Name of the language model to use.
                Defaults to "gpt-4o-mini".
            max_tokens (int): Maximum number of tokens for model output.
                Defaults to 1024.
            json_mode (bool): Whether to use JSON mode (response_format) for structured output.
                Defaults to True (preserves backward compatibility with OpenAI).
            base_url (str | None): Base URL for vLLM server. Used when llm_api="vllm".
                Defaults to None.
            api_key (str | None): API key for vLLM server. Used when llm_api="vllm".
                Defaults to None.
            num_runs (int): Number of times to run NER and take the union of results.
                Increases robustness when the LLM is non-deterministic. Defaults to 1.
        """

        self.llm_api = llm_api
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.num_runs = max(1, num_runs)

        self.client = init_langchain_model(
            llm_api,
            model_name,
            base_url=base_url,
            api_key=api_key,
        )

    def _call_once(self, text: str) -> list:
        query_ner_prompts = ChatPromptTemplate.from_messages(
            [
                SystemMessage("You're a very effective entity extraction system."),
                HumanMessage(query_prompt_one_shot_input),
                AIMessage(query_prompt_one_shot_output),
                HumanMessage(query_prompt_template.format(text)),
            ]
        )
        query_ner_messages = query_ner_prompts.format_prompt()

        json_mode_used = False
        if self.json_mode:  # JSON mode
            chat_completion = self.client.invoke(
                query_ner_messages.to_messages(),
                temperature=0,
                max_tokens=self.max_tokens,
                stop=["\n\n"],
                response_format={"type": "json_object"},
            )
            response_content = chat_completion.content
            chat_completion.response_metadata["token_usage"]["total_tokens"]
            json_mode_used = True
        elif isinstance(self.client, ChatOllama) or isinstance(
            self.client, ChatLlamaCpp
        ):
            response_content = self.client.invoke(query_ner_messages.to_messages())
            if hasattr(response_content, "content"):
                response_content = response_content.content
            response_content = extract_json_dict(response_content)
        else:  # no JSON mode
            chat_completion = self.client.invoke(
                query_ner_messages.to_messages(),
                temperature=0,
                max_tokens=self.max_tokens,
                stop=["\n\n"],
            )
            response_content = chat_completion.content
            response_content = extract_json_dict(response_content)
            chat_completion.response_metadata["token_usage"]["total_tokens"]

        if not json_mode_used:
            try:
                assert "named_entities" in response_content
                response_content = str(response_content)
            except Exception as e:
                print("Query NER exception", e)
                response_content = {"named_entities": []}

        try:
            import json

            ner_list = json.loads(response_content)["named_entities"]
            return [processing_phrases(ner) for ner in ner_list]
        except Exception as e:
            logger.error(f"Error in extracting named entities: {e}")
            return []

    def __call__(self, text: str) -> list:
        """Process text input to extract named entities using different chat models.

        Runs NER `num_runs` times and returns the union of results (preserving first-seen order).
        Multiple runs improve robustness when the LLM is non-deterministic at temperature=0.

        Args:
            text (str): The input text to extract named entities from.

        Returns:
            list: A list of processed named entities extracted from the text.
                 Returns empty list if extraction fails.
        """
        seen: dict[str, None] = {}
        for _ in range(self.num_runs):
            for entity in self._call_once(text):
                seen[entity] = None
        return list(seen.keys())
