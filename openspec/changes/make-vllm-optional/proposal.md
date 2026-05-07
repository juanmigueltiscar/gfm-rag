## Why

`vllm` es una dependencia pesada (requiere CUDA toolkit en build-time, compila extensiones C++, incompatible con ARM64) que actualmente es obligatoria en `pyproject.toml` pero solo se usa en un modelo: `Qwen3TextEmbModel`. El proyecto tiene alternativas que no requieren vllm (`BaseTextEmbModel`, `NVEmbedV2`), pero la importación a nivel de módulo en `text_emb_models/__init__.py` fuerza a todos los usuarios a instalar vllm incluso si usan otro modelo de embeddings. Esto bloquea el uso del proyecto en arquitecturas ARM64 (Apple Silicon, AWS Graviton) y añade complejidad innecesaria en entornos sin GPU.

## What Changes

- Mover `vllm>=0.11.0` de `[project].dependencies` a un grupo opcional `[project.optional-dependencies].vllm`
- Hacer lazy/condicional el import de `Qwen3TextEmbModel` en `text_emb_models/__init__.py` para que no falle si vllm no está instalado
- Añadir un `try/except ImportError` en `qwen3_model.py` para dar un mensaje claro si se intenta usar sin vllm instalado
- Documentar la instalación opcional (`pip install gfmrag[vllm]` o `uv sync --group vllm`)

## Capabilities

### New Capabilities

- `optional-vllm-dependency`: El paquete vllm es una dependencia opcional para el modelo de embeddings Qwen3, no una dependencia obligatoria del proyecto

### Modified Capabilities

<!-- No se modifican specs existentes. Las specs `langchain-vllm-backend` y `llm-vllm-backend` usan la API REST de un servidor vLLM (vía `ChatOpenAI`), no el paquete Python `vllm`. -->

## Impact

- `pyproject.toml`: mover vllm a grupo opcional
- `gfmrag/text_emb_models/__init__.py`: import condicional de Qwen3TextEmbModel
- `gfmrag/text_emb_models/qwen3_model.py`: `ImportError` descriptivo
- `README.md` / docs: instrucciones de instalación opcional actualizadas (si aplica)
- `AGENTS.md`: nota sobre dependencia opcional (si aplica)

## Non-goals

- No se modifica `VLLMModel` ni `init_langchain_model()` — ambos usan REST API, no el paquete vllm
- No se elimina `Qwen3TextEmbModel` — sigue funcionando igual si vllm está instalado
- No se toca el comportamiento de NER/OpenIE con backend vllm
