## Why

El proyecto actualmente solo puede usar modelos de lenguaje remotos a través de APIs propietarias (OpenAI, Gemini, Mistral) o modelos locales vía HuggingFace transformers. No existe soporte para servidores de inferencia compatibles con OpenAI que ejecuten modelos open-source como Llama, Mistral o Qwen — el caso de uso más común en entornos de producción on-premise. vLLM es el servidor de inferencia de referencia para este propósito y expone una API OpenAI-compatible, lo que minimiza la fricción de integración. El proyecto ya usa este patrón en la capa de embeddings (`Qwen3TextEmbModel`) pero no en la capa de LLM ni en la capa de construcción de grafos (NER/OpenIE).

## What Changes

- Nueva clase `VLLMModel` en `gfmrag/llms/` implementando `BaseLanguageModel`, que se comunica con un servidor vLLM usando el SDK de OpenAI con `base_url` configurable
- Tokenización a través del endpoint `/tokenize` de vLLM (consistencia exacta con el servidor)
- Health check estricto al inicializar (lanza `RuntimeError` si el servidor no responde)
- Nuevo branch `"vllm"` en la factoría `init_langchain_model()` de `langchain_util.py` para la capa de construcción de grafos
- Nueva bandera `json_mode` en `LLMNERModel` y `LLMOPENIEModel` para desacoplar la decisión de usar JSON mode del tipo concreto de cliente LangChain, añadiendo también `base_url` y `api_key` como parámetros configurables
- Nuevos Hydra configs de ejemplo para QA/IRCoT con `_target_: gfmrag.llms.VLLMModel` y para NER/OpenIE con `llm_api: vllm`

### Non-goals

- No se modifica la implementación existente de `ChatGPT` ni sus configuraciones YAML
- No se añade soporte para arrancar un servidor vLLM desde el proyecto (el servidor debe estar ya corriendo)
- No se modifica `HfCausalModel` (modelos locales con transformers)
- No se modifica la capa de embeddings (`Qwen3TextEmbModel`), que ya tiene su propio soporte vLLM

## Capabilities

### New Capabilities

- `llm-vllm-backend`: Capacidad de usar un modelo servido por vLLM como backend de generación de texto en los workflows de QA e IRCoT, reemplazando `ChatGPT`. Incluye health check, tokenización vía endpoint `/tokenize`, auto-detección de `max_model_len`, y retry con backoff.
- `langchain-vllm-backend`: Capacidad de usar un modelo servido por vLLM como backend LLM en la construcción de grafos (NER y OpenIE) a través de LangChain. El cliente devuelto es `ChatOpenAI` con `base_url` configurable. Incluye soporte para deshabilitar JSON mode cuando el servidor vLLM no tiene guided decoding.
- `ner-json-mode-flag`: Nueva bandera `json_mode` en `LLMNERModel` y `LLMOPENIEModel` que desacopla la decisión de usar JSON mode estructurado del tipo concreto de cliente LangChain, permitiendo que backends compatibles con OpenAI pero sin soporte de guided decoding funcionen correctamente.

### Modified Capabilities

<!-- No se modifican specs existentes. -->

## Impact

- **Nuevo archivo**: `gfmrag/llms/vllm_model.py`
- **Modificado**: `gfmrag/llms/__init__.py` — añadir `VLLMModel` a `__all__`
- **Modificado**: `gfmrag/graph_index_construction/langchain_util.py` — añadir branch `"vllm"` a la factoría, con parámetros `base_url` y `api_key`
- **Modificado**: `gfmrag/graph_index_construction/ner_model/llm_ner_model.py` — añadir `json_mode`, `base_url`, `api_key` params; cambiar `isinstance` checks a `self.json_mode`
- **Modificado**: `gfmrag/graph_index_construction/openie_model/llm_openie_model.py` — añadir `json_mode`, `base_url`, `api_key` params; cambiar `isinstance` checks a `self.json_mode`
- **Modificado**: `gfmrag/workflow/config/ner_model/llm_ner_model.yaml` — añadir `json_mode: true` (default preserva comportamiento actual)
- **Modificado**: `gfmrag/workflow/config/openie_model/llm_openie_model.yaml` — añadir `json_mode: true` (default preserva comportamiento actual)
- **Nuevas dependencias**: Ninguna. `openai` (SDK) y `requests` ya son dependencias del proyecto. `ChatOpenAI` de `langchain-openai` ya se usa.
- **Sin cambios en**: `ChatGPT`, `HfCausalModel`, `Gemini`, `MistralSmall32API`, capa de embeddings, configuraciones existentes de QA/IRCoT
