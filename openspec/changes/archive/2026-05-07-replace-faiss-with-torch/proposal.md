## Why

`faiss-gpu-cu12` no tiene wheels para `aarch64` (solo `x86_64`), lo que bloquea la instalación del proyecto en máquinas ARM64 como la actual (NVIDIA GB10 Grace Blackwell). Además, el código solo usa `IndexFlatIP` (búsqueda exacta por producto interno), una operación que no se beneficia de GPU y que puede implementarse trivialmente con `torch`, que ya es dependencia del proyecto. En paralelo, `torch 2.8.0` en `aarch64` no incluye soporte CUDA (101 MB, CPU-only), mientras que `torch >=2.9.0` sí lo hace (420 MB con paquetes `nvidia-*` para ARM64), lo que habilita el uso de la GPU disponible.

## What Changes

- **Eliminar** `faiss-gpu-cu12` como dependencia
- **Reemplazar** `IndexFlatIP` y `faiss.normalize_L2` por operaciones equivalentes en `torch` dentro de `HippoRAG2Constructor` (único consumidor de FAISS)
- **Subir** `torch` de `>=2.4.1` a `>=2.9.0` (efectivamente 2.11.0) para habilitar CUDA en `aarch64`
- **Re-generar** `uv.lock` con las nuevas versiones

## Capabilities

### New Capabilities

- `vector-search-torch`: Búsqueda de vecinos por producto interno usando torch en lugar de FAISS, incluyendo normalización L2 y construcción de índice flat.

### Modified Capabilities

Ninguna. Los specs existentes (`langchain-vllm-backend`, `llm-vllm-backend`, `ner-json-mode-flag`) no se ven afectados.

## Impact

- **Dependencias**: Se elimina `faiss-gpu-cu12`, se sube cota inferior de `torch` a `>=2.9.0`
- **Código**: Solo `gfmrag/graph_index_construction/sft_constructors/hipporag2_constructor.py` (~30 líneas)
- **API pública**: Sin cambios — `HippoRAG2Constructor` mantiene exactamente la misma interfaz y comportamiento
- **Rendimiento**: Idéntico para `IndexFlatIP` (mismo algoritmo O(n·d)); potencialmente mejor si torch opera en GPU
- **Breaking**: Ninguno

## Non-goals

- No se migran otros constructores (solo `HippoRAG2Constructor` usa FAISS)
- No se cambia el algoritmo de búsqueda (sigue siendo fuerza bruta exacta, no aproximada)
- No se añaden índices aproximados (IVF, HNSW, etc.)
- No se modifica la API de `BaseTextEmbModel` ni el pipeline de embeddings
