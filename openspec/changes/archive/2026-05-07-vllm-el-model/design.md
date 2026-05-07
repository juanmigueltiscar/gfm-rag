## Context

El proyecto dispone de una jerarquía clara para entity linking: `BaseELModel` como ABC, con implementaciones concretas (`DPRELModel`, `NVEmbedV2ELModel`, `ColbertELModel`). Todas las implementaciones actuales cargan el modelo localmente. En paralelo, `Qwen3TextEmbModel` ya demuestra cómo conectarse a un servidor vLLM remoto via OpenAI-compatible API, pero vive en la capa de `text_emb_models` y no implementa `BaseELModel`.

El usuario tiene `nvidia/llama-nemotron-embed-1b-v2` servido en vLLM y quiere utilizarlo como modelo de entity linking sin duplicar la instancia del modelo.

## Goals / Non-Goals

**Goals:**
- Implementar `VLLMELModel(BaseELModel)` que delega los embeddings a un servidor vLLM remoto via `/v1/embeddings`.
- Mantener la misma interfaz (`index()` / `__call__()`) que el resto de EL models.
- Soportar `query_instruct` / `passage_instruct`, normalización L2 y caché en disco.
- Añadir un fichero de configuración Hydra listo para usar.
- No requerir el paquete `vllm` instalado localmente.

**Non-Goals:**
- Arrancar un servidor vLLM local (fuera de scope; lo hace `Qwen3TextEmbModel`).
- Modificar interfaces existentes.
- Soporte async/streaming.

## Decisions

### D1: Clase autónoma en `entity_linking_model/`, no wrapper de `Qwen3TextEmbModel`

**Decisión**: `VLLMELModel` implementa directamente `BaseELModel` y replica solo la lógica de cliente OpenAI necesaria.

**Alternativa descartada**: Crear un adapter `TextEmbModelELAdapter(BaseELModel)` que acepte cualquier `BaseTextEmbModel`. Esto sería más genérico, pero introduce una abstracción extra innecesaria para el caso de uso actual y hace más compleja la configuración Hydra (objetos anidados).

**Rationale**: La coherencia con `DPRELModel` (estructura paralela) facilita la mantenibilidad. Si en el futuro se necesita el adapter, se puede refactorizar sin romper la API pública.

### D2: Solo modo remoto (servidor ya activo)

**Decisión**: `VLLMELModel` asume que el servidor vLLM está corriendo. Lanza `RuntimeError` si el health check falla en `__init__`.

**Alternativa descartada**: Arrancar servidor local (como `Qwen3TextEmbModel`). No aplica porque el caso de uso es reutilizar infraestructura existente.

### D3: Health check en `__init__` via `/health` endpoint

**Decisión**: Igual que `Qwen3TextEmbModel._is_api_available()`, se hace un GET a `{api_base_without_v1}/health` antes de crear el cliente.

**Rationale**: Fallo rápido con mensaje claro mejor que error críptico en la primera llamada de embeddings.

### D4: Caché en disco con MD5 fingerprint

**Decisión**: Reutilizar el mismo patrón de caché de `DPRELModel`: `{root}/{model_name_sanitized}_vllm_cache/{md5_of_entity_list}.pt`.

**Rationale**: Consistencia con el resto del código. Los embeddings de entidades cambian raramente; la caché evita llamadas de red innecesarias en reruns.

### D5: Normalización L2 opcional post-recuperación

**Decisión**: Normalizar en cliente (con `torch.nn.functional.normalize`) en lugar de depender de que el servidor la aplique.

**Rationale**: vLLM no garantiza que todos los modelos soporten el parámetro `encoding_format` con normalización. Más portable.

## Risks / Trade-offs

- **[Risk] El servidor vLLM está caído durante `index()`** → El error de red se propaga tal cual. Mitigación: el health check en `__init__` avisa antes de comenzar el indexado.
- **[Risk] Latencia de red vs. inferencia local** → Para conjuntos de entidades grandes, el batching secuencial puede ser más lento que `SentenceTransformer` local con GPU. Mitigación: el parámetro `batch_size` permite ajustar el throughput; la caché elimina el coste en reruns.
- **[Trade-off] Duplicación de lógica de cliente OpenAI** → `Qwen3TextEmbModel` y `VLLMELModel` comparten código similar para llamar a `/v1/embeddings`. Aceptable por ahora; si crece, se puede extraer a un módulo `_openai_embed_client.py` compartido.

## Migration Plan

No hay migración: es funcionalidad nueva. Los flujos existentes con `DPRELModel` no se ven afectados. Para adoptar `VLLMELModel`, el usuario solo necesita cambiar el YAML de configuración `el_model`.

## Open Questions

- ¿El modelo `nvidia/llama-nemotron-embed-1b-v2` requiere algún instruct prefix especial para queries vs. passages? (pendiente de confirmar con el usuario; el campo `query_instruct` permite configurarlo sin cambios de código.)
