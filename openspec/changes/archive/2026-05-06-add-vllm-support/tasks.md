## 1. Capa gfmrag/llms/ — VLLMModel

- [x] 1.1 Crear `gfmrag/llms/vllm_model.py` con la clase `VLLMModel(BaseLanguageModel)`: constructor con health check (`_is_api_available` vía `GET /health`), parámetros `model_name_or_path`, `base_url`, `api_key`, `retry`, `timeout`, `maximun_token`, `temperature`. Si `maximun_token=None`, auto-detectar vía `GET /v1/models`.
- [x] 1.2 Implementar `token_len()` usando `POST /tokenize` de vLLM con fallback a `AutoTokenizer.from_pretrained()` si el endpoint devuelve 404.
- [x] 1.3 Implementar `generate_sentence()` usando `client.chat.completions.create()` con truncation de input (mismo patrón que `ChatGPT`) y retry con backoff de 30 segundos.
- [x] 1.4 Registrar `VLLMModel` en `gfmrag/llms/__init__.py` (`__all__` y export).

## 2. Capa langchain_util.py — Soporte vLLM

- [x] 2.1 Añadir branch `"vllm"` en `init_langchain_model()` en `langchain_util.py`. Aceptar `base_url` y `api_key` como kwargs. Devolver `ChatOpenAI(openai_api_base=base_url, openai_api_key=api_key, model=model_name, temperature=temperature, max_retries=max_retries, timeout=timeout, **kwargs)`.

## 3. Capa NER/OpenIE — Bandera json_mode y parámetros vLLM

- [x] 3.1 Añadir parámetros `json_mode: bool = True`, `base_url: str = None`, `api_key: str = None` al constructor de `LLMNERModel`. Forwardear `base_url` y `api_key` a `init_langchain_model()`.
- [x] 3.2 Reemplazar `isinstance(self.client, ChatOpenAI)` por `self.json_mode` en `LLMNERModel.__call__()`. Mismo comportamiento: `json_mode=True` usa `response_format={"type": "json_object"}`, `json_mode=False` usa `extract_json_dict()`.
- [x] 3.3 Añadir parámetros `json_mode: bool = True`, `base_url: str = None`, `api_key: str = None` al constructor de `LLMOPENIEModel`. Forwardear `base_url` y `api_key` a `init_langchain_model()`.
- [x] 3.4 Reemplazar `isinstance(self.client, ChatOpenAI)` por `self.json_mode` en ambos métodos `ner()` y `openie_post_ner_extract()` de `LLMOPENIEModel`. Mismo comportamiento que en NER.
- [x] 3.5 Actualizar los `Literal` type hints de `llm_api` en ambos modelos para incluir `"vllm"`.

## 4. Hydra Configs

- [x] 4.1 Añadir `json_mode: true` a `gfmrag/workflow/config/ner_model/llm_ner_model.yaml` (default preserva comportamiento actual).
- [x] 4.2 Añadir `json_mode: true` a `gfmrag/workflow/config/openie_model/llm_openie_model.yaml` (default preserva comportamiento actual).

## 5. Verificación

- [x] 5.1 Ejecutar ruff lint + mypy typecheck vía `pre-commit run --all-files --show-diff-on-failure`.
- [x] 5.2 Verificar que los imports existentes de `gfmrag.llms` no se rompen (`ChatGPT`, `HfCausalModel`, `BaseLanguageModel` siguen exportados).
- [x] 5.3 Verificar que los configs YAML existentes de QA/IRCoT/NER/OpenIE siguen funcionando sin cambios (backward compatibility).
