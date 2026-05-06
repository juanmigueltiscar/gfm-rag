# Aportaciones Gaiatec sobre upstream GFM-RAG v2.0.0

Análisis de los archivos modificados respecto al upstream v2.0.0, tras `gaiatec` = `upstream/main`.

---

## A. Archivos en mismo path (diff directo)

### 1. `.gitignore`
El usuario añade `.venv/`, `GFM-RAG-8M`, `models/`, `log.txt`. El upstream tiene además `data` y `/wandb/` que el usuario no ignora.

### 2. `.pre-commit-config.yaml`
El usuario quitó el hook `poetry-check` porque usa `uv` como gestor de dependencias. Versiones de hooks ligeramente distintas (ruff v0.9.10 vs v0.12.0, etc.).

### 3. `gfmrag/llms/__init__.py`
El usuario añade la exportación de `Gemini`. El upstream la quitó.

```python
# Usuario
from .gemini import Gemini
__all__ = ["BaseLanguageModel", "HfCausalModel", "ChatGPT", "Gemini"]

# Upstream
__all__ = ["BaseLanguageModel", "HfCausalModel", "ChatGPT"]
```

### 4. `gfmrag/gfmrag_retriever.py` — REESCRITO COMPLETO en v2.0.0

| Qué añadía el usuario (v1.0.0) | Estado en upstream v2.0.0 |
|---|---|
| `retrieve_batch()` — procesamiento batch asíncrono con asyncio | No existe |
| `retrieve()` devuelve `(docs, entidades)` — docs + entidades NER | `retrieve()` devuelve `dict[str, list[dict]]` tipado por tipo de nodo |
| `prepare_input_for_graph_retriever()` devuelve `(input, entities)` | Solo devuelve `dict` |
| `from_config()` factory | Reemplazado por `from_index()` |
| Usa `QADataset`, `doc_ranker`, `doc_retriever`, `ent2id`, `entities_weight` | Usa `GraphIndexDataset`, `node_info`, `node2id`, `start_nodes_mask` |
| Imports desde `gfmrag.kg_construction.*` | Imports desde `gfmrag.graph_index_construction.*` |

### 5. `gfmrag/utils/util.py`

| Qué añadía el usuario | Estado en upstream v2.0.0 |
|---|---|
| `strict=False` en `load_state_dict` (compatibilidad entre arquitecturas) | `strict=True` (sin tolerancia) |
| `get_entities_weight()` — pesos por frecuencia de entidad | Eliminada del upstream |
| `get_multi_dataset()` (nombre antiguo) | Renombrada a `init_multi_dataset()` + `check_all_files_exist()` |

### 6. `pyproject.toml`
El usuario migró de Poetry a **uv** (gestor de paquetes). Dependencias distintas:
- Usuario eliminó `colbert-ai`/`ragatouille`, `faiss-gpu-cu12`
- Usuario añadió `google-generativeai`, `ipykernel`, `einops`, `powertools`
- Upstream añadió `vllm`, `pymetis`, `pylate`
- Upstream usa `[tool.poetry]` + `poetry-core`, usuario usa `[project]` + `hatchling`

---

## B. Archivos renombrados (`kg_construction` → `graph_index_construction`)

### 7. `langchain_util.py`

| Aportación del usuario | Estado en upstream |
|---|---|
| Soporte `google` → `ChatGoogleGenerativeAI` | No existe |
| Servidor OpenAI-compatible local (`base_url`, `api_key=""`) | No existe; upstream fuerza `api_key` real + `assert model_name.startswith("gpt-")` |
| Quita el `assert` que fuerza modelos "gpt-" | Upstream lo mantiene |
| Sin `n_ctx` ni `low_vram` | Upstream los añadió para llama.cpp/ollama |

### 8. `ner_model/llm_ner_model.py`

| Aportación del usuario | Estado en upstream |
|---|---|
| **Prompts en español** (system message, one-shot example, templates) | Inglés |
| **`acall()`** — método asíncrono para NER concurrente con `asyncio.gather()` | No existe |
| Soporte `ChatGoogleGenerativeAI` como backend NER | Eliminado |
| Prints de debug de respuestas crudas del LLM | Eliminados |

### 9. `utils.py`

| Aportación del usuario | Estado en upstream |
|---|---|
| `processing_phrases()` preserva **acentos y caracteres españoles** (ÁÉÍÓÚÜÑáéíóúüñ) en regex | Solo `[^A-Za-z0-9 ]` (ASCII) |
| Sin `generate_uuid()` | Upstream añadió `generate_uuid()` |

### 10. `colbert_el_model.py` — REESCRITO COMPLETO

| Usuario (v1.0.0) | Upstream (v2.0.0) |
|---|---|
| Usaba `colbert-ai` (ColBERT, Indexer, Searcher) | Usa `pylate` (ColBERT, PLAID index) + qdrant |
| API completamente distinta | API nueva |

---

## C. Configs de workflow (stage*.yaml) — ARCHIVOS ROTOS

Los 5 archivos de config contienen marcadores de conflicto de merge sin resolver. **No son YAMLs válidos.** En v2.0.0 se movieron a `gfm_rag/` y se renombraron (sin prefijo `stage*_`).

| Backup (roto) | Nuevo path en v2.0.0 |
|---|---|
| `stage1_index_dataset.yaml` | `gfm_rag/index_dataset.yaml` |
| `stage2_qa_finetune.yaml` | `gfm_rag/sft_training.yaml` |
| `stage3_qa_inference.yaml` | `gfm_rag/qa_inference.yaml` |
| `stage3_qa_ircot_inference.yaml` | `gfm_rag/qa_ircot_inference.yaml` |
| `exp_visualize_path.yaml` | `gfm_rag/exp_visualize_path.yaml` |

Aportaciones clave en las configs del usuario:
- `root: ../../bbdd` (rutas relativas al proyecto)
- `el_model: itc_nomic_ai` / `itc_bge`
- `agent_prompt: itc_ircot`, `qa_prompt: itc`
- `llm: Gemini` con `gemini-2.5-pro`
- `num_epoch: 60`, `do_eval: yes`

---

## Resumen para incorporación

| Nivel | Archivos | Acción recomendada |
|---|---|---|
| **Fácil (merge limpio)** | `.gitignore`, `.pre-commit-config.yaml` | Añadir líneas del usuario sobre la versión upstream |
| **Medio (merge con cuidado)** | `langchain_util.py`, `ner_model/llm_ner_model.py`, `utils.py`, `llms/__init__.py` | Portar las funcionalidades del usuario preservando las mejoras del upstream |
| **Difícil (rehacer)** | `gfmrag_retriever.py`, `colbert_el_model.py` | La interfaz cambió totalmente. Reimplementar `retrieve_batch()` y retorno de entidades sobre la nueva API |
| **Rehacer desde cero** | `pyproject.toml`, `util.py`, `stage*.yaml` | Diferencias estructurales (poetry vs uv), o archivos rotos con conflictos |
