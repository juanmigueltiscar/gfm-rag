## Context

Actualmente `vllm>=0.11.0` está en `[project].dependencies` de `pyproject.toml`, lo que lo convierte en una dependencia obligatoria. Sin embargo, el paquete `vllm` solo se usa en un lugar: `Qwen3TextEmbModel` para servir modelos de embeddings Qwen3 localmente. El proyecto tiene otros modelos de embeddings (`BaseTextEmbModel` con SentenceTransformers, `NVEmbedV2` con HuggingFace) que no requieren vllm.

El problema es que `gfmrag/text_emb_models/__init__.py` hace `from .qwen3_model import Qwen3TextEmbModel`, y `qwen3_model.py` tiene `from vllm import LLM, PoolingParams` a nivel de módulo. Esto significa que cualquier import de `gfmrag.text_emb_models` (que ocurre en el flujo principal de retrieval e indexación) fuerza la importación de vllm, fallando si no está instalado.

NOTA: Los modelos `VLLMModel` (en `gfmrag/llms/`) y el backend vllm en NER/OpenIE (en `graph_index_construction/`) NO importan el paquete `vllm`. Usan la API REST OpenAI-compatible de un servidor vLLM externo. Estos no se ven afectados.

## Goals / Non-Goals

**Goals:**
- Permitir que el proyecto funcione sin tener instalado el paquete `vllm`
- Mantener `Qwen3TextEmbModel` funcional si vllm está instalado
- Dar un mensaje de error claro si se intenta usar `Qwen3TextEmbModel` sin vllm
- Mover vllm a un grupo de dependencias opcional en `pyproject.toml`

**Non-Goals:**
- No modificar `VLLMModel` ni el backend `langchain-vllm` — ya funcionan sin el paquete vllm
- No eliminar `Qwen3TextEmbModel`
- No añadir nuevas funcionalidades de embeddings
- No cambiar la API pública de `text_emb_models`

## Decisions

### Decisión 1: Lazy import en `__init__.py` vs lazy import en `qwen3_model.py`

**Elegido**: Lazy import en `__init__.py` + `ImportError` descriptivo en `qwen3_model.py`.

**Alternativas consideradas**:
- Mover el import a nivel de función dentro de `__init__.py`: requeriría un factory o función `get_qwen3_model()`, rompiendo la API actual donde Hydra instancia `Qwen3TextEmbModel` vía `_target_`.
- Envolver `__init__.py` en try/except: demasiado silencioso, el error aparecería tarde y de forma confusa.
- Plugin system / entry points: sobre-ingeniería para un solo modelo.

**Implementación**: Mover el `from .qwen3_model import Qwen3TextEmbModel` dentro de una función `_import_qwen3()` llamada lazy. Pero la forma más limpia y que no rompe Hydra es mantener el import en `__init__.py` con un try/except que capture `ImportError`:

```python
# gfmrag/text_emb_models/__init__.py
from .base_model import BaseTextEmbModel
from .nv_embed import NVEmbedV2

try:
    from .qwen3_model import Qwen3TextEmbModel
except ImportError:
    Qwen3TextEmbModel = None  # type: ignore

__all__ = ["BaseTextEmbModel", "NVEmbedV2", "Qwen3TextEmbModel"]
```

Y en `qwen3_model.py`, cambiar el import absoluto por uno con mensaje descriptivo:

```python
# gfmrag/text_emb_models/qwen3_model.py
try:
    from vllm import LLM, PoolingParams
except ImportError:
    raise ImportError(
        "vllm is required for Qwen3TextEmbModel. "
        "Install it with: pip install gfmrag[vllm] or uv sync --group vllm"
    )
```

**IMPORTANTE**: Hydra's `instantiate()` con `_target_: gfmrag.text_emb_models.Qwen3TextEmbModel` seguirá funcionando porque el módulo `qwen3_model` se importa en ese momento (cuando se necesita), y para entonces el usuario ya debería tener vllm instalado si quiere usar Qwen3. El `ImportError` dará un mensaje claro.

### Decisión 2: Nombre del grupo opcional

**Elegido**: `[project.optional-dependencies].vllm` (grupo `vllm`).

**Alternativas consideradas**:
- `embedding-vllm`: más descriptivo pero verboso
- `qwen3`: no deja claro que es la dependencia vllm

El nombre `vllm` es directo y sigue la convención de `pip install package[extra]`.

### Decisión 3: Formato de pyproject.toml

**Elegido**: Usar `[project.optional-dependencies]` estándar de PEP 621.

Esto permite `pip install gfmrag[vllm]` y es compatible con `uv sync --group vllm` (uv mapea optional-dependencies a dependency groups automáticamente).

## Riesgos / Trade-offs

- **[Riesgo]**: Si alguien tiene `vllm` instalado pero en una versión incompatible, el `try/except ImportError` en `__init__.py` capturará el error y `Qwen3TextEmbModel` será `None`. Esto podría causar un `AttributeError` confuso más tarde.
  → **Mitigación**: El `ImportError` en `qwen3_model.py` solo se lanza cuando se intenta importar el módulo (al instanciar la clase). El `try/except` en `__init__.py` solo silencia el error de import del módulo, no de la clase.

- **[Riesgo]**: Tests que referencian `Qwen3TextEmbModel` vía `_target_` en Hydra fallarán si vllm no está instalado en CI.
  → **Mitigación**: Los tests que necesitan Qwen3 deben marcar `vllm` como dependencia de test o usar skip. Los tests existentes que usan `BGETextEmbModel` (test_gfmrag_retriever.py) no se ven afectados.

- **[Trade-off]**: `Qwen3TextEmbModel` aparece en `__all__` aunque puede ser `None`. Esto es aceptable porque `__all__` es una convención de exportación, no una garantía de instanciabilidad. La alternativa sería una función `has_qwen3_support()` pero añade complejidad innecesaria.
