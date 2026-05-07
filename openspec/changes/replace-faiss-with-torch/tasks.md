## 1. Dependencias

- [ ] 1.1 Reemplazar `faiss-gpu-cu12>=1.9.0.post1,<2.0.0` por `torch>=2.9.0` en `pyproject.toml`
- [ ] 1.2 Ejecutar `uv lock` para regenerar `uv.lock` sin FAISS y con torch 2.11.0+ y paquetes `nvidia-*` para aarch64
- [ ] 1.3 Ejecutar `uv sync` para instalar el entorno actualizado

## 2. Migración del código

- [ ] 2.1 Reemplazar `faiss.normalize_L2(emb)` por `torch.nn.functional.normalize(emb, p=2, dim=1)` en `_encode_texts`
- [ ] 2.2 Reemplazar `faiss.IndexFlatIP` por `torch.Tensor` en `_build_faiss_index`: devolver el tensor de embeddings en lugar de crear un índice FAISS
- [ ] 2.3 Reemplazar `index.search(query, k)` por `torch.mm` + `torch.topk` en `_search_by_type` y `retrieve_fact_candidates`, usando `index_tensor.shape[0]` en lugar de `index.ntotal`
- [ ] 2.4 Actualizar type hints: `dict[str, faiss.IndexFlatIP]` → `dict[str, torch.Tensor]` y `faiss.IndexFlatIP | None` → `torch.Tensor | None`
- [ ] 2.5 Eliminar `import faiss` y `import numpy as np` (si numpy no se usa en otro lado del archivo)
- [ ] 2.6 Verificar que `torch` ya está importado en el archivo (está en línea 12)
- [ ] 2.7 Añadir `import torch.nn.functional as F` donde sea necesario

## 3. Verificación

- [ ] 3.1 Ejecutar `pre-commit run --all-files --show-diff-on-failure` y corregir errores
- [ ] 3.2 Ejecutar tests relevantes: `uv run python -m pytest tests/ -k "hipporag" -v`
- [ ] 3.3 Verificar que `torch.cuda.is_available()` devuelve `True` en la máquina actual
- [ ] 3.4 Ejecutar `uv run python -m pytest tests/ -v` (todos los tests) para detectar regresiones
