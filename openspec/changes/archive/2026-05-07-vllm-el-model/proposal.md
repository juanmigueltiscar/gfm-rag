## Why

El sistema de entity linking actual carga los modelos de embedding localmente via `SentenceTransformer`, lo que impide reutilizar modelos ya servidos en infraestructura vLLM existente (p.ej. `nvidia/llama-nemotron-embed-1b-v2`). Esto obliga a mantener dos instancias del mismo modelo y consume memoria GPU innecesaria.

## What Changes

- **Nuevo modelo `VLLMELModel`** en `gfmrag/graph_index_construction/entity_linking_model/vllm_el_model.py`: implementa `BaseELModel` usando el cliente OpenAI para llamar a `/v1/embeddings` en un servidor vLLM remoto.
- **Registro en `__init__.py`** del módulo `entity_linking_model` para exponer `VLLMELModel`.
- **Config Hydra** `gfmrag/workflow/config/el_model/vllm_el_model.yaml` con los parámetros del nuevo modelo.
- Sin nuevas dependencias de paquete: usa `openai` (ya presente) en lugar del paquete `vllm`.

## Capabilities

### New Capabilities

- `vllm-el-model`: Entity linking via embeddings servidos en un servidor vLLM remoto accesible a través de la API OpenAI-compatible (`/v1/embeddings`). Incluye soporte para instrucciones de query/passage, normalización L2, y caché de embeddings en disco.

### Modified Capabilities

## Impact

- **Código afectado**: `gfmrag/graph_index_construction/entity_linking_model/` (nuevo fichero + `__init__.py`)
- **Configuración**: nuevo YAML en `gfmrag/workflow/config/el_model/`
- **Dependencias**: ninguna nueva; `openai` ya es dependencia del proyecto
- **APIs**: ningún cambio en interfaces existentes; `VLLMELModel` implementa `BaseELModel` sin modificarla
- **Tests**: nuevo test en `tests/` para cubrir inicialización y llamadas al servidor

## Non-goals

- No se soporta arrancar un servidor vLLM local desde `VLLMELModel` (eso ya existe en `Qwen3TextEmbModel`; este modelo asume servidor externo ya activo).
- No se modifican los modelos existentes (`DPRELModel`, `NVEmbedV2ELModel`, `ColbertELModel`).
- No se añade soporte para autenticación con tokens Bearer u OAuth (se acepta `api_key` como string simple, igual que el resto del proyecto).
- No se soporta modo asíncrono/batch paralelo más allá del batching secuencial ya presente en `Qwen3TextEmbModel`.
