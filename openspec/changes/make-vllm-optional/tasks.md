## 1. Dependencies

- [x] 1.1 Mover `vllm>=0.11.0,<0.12.0` de `[project].dependencies` a `[project.optional-dependencies].vllm` en `pyproject.toml`
- [x] 1.2 Verificar que `uv sync` instala el resto de dependencias sin vllm
- [ ] 1.3 Verificar que `uv sync --group vllm` instala vllm correctamente

## 2. Core Implementation

- [x] 2.1 Añadir `try/except ImportError` con mensaje descriptivo en `gfmrag/text_emb_models/qwen3_model.py` alrededor del `from vllm import LLM, PoolingParams`
- [x] 2.2 Cambiar `from .qwen3_model import Qwen3TextEmbModel` por un import condicional con `try/except ImportError` en `gfmrag/text_emb_models/__init__.py`, asignando `None` si falla
- [x] 2.3 Verificar que `from gfmrag.text_emb_models import BaseTextEmbModel, NVEmbedV2` funciona sin vllm instalado
- [x] 3.1 Ejecutar tests existentes que usan `BGETextEmbModel` (en `test_gfmrag_retriever.py`) sin vllm instalado para confirmar que no se rompen
- [x] 3.2 Ejecutar `pre-commit run --all-files` para verificar que linting/formato pasan
- [x] 3.3 Probar la importación de `Qwen3TextEmbModel` sin vllm: confirmar que `ImportError` contiene mensaje con instrucciones de instalación
- [ ] 3.4 (Opcional, solo si hay acceso a GPU) Probar `Qwen3TextEmbModel` con vllm instalado para confirmar que funciona igual que antes
