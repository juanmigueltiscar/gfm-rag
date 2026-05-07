## Why

Poetry cumple, pero uv es significativamente más rápido (resolución e instalación), está desarrollado por Astral (misma empresa que Ruff, que ya usamos), y unifica el tooling de Python bajo un solo binario. Migrar reduce la complejidad del entorno de desarrollo y acelera los CI locales y remotos.

## What Changes

- `pyproject.toml`: reescribir secciones de build y dependencias al formato PEP 621, con `hatchling` como build backend
- `poetry.lock` → `uv.lock`: regenerar lockfile con `uv lock`
- `.pre-commit-config.yaml`: eliminar hook `poetry-check`
- CI/CD (3 workflows de GitHub Actions): reemplazar `snok/install-poetry` por `astral-sh/setup-uv`, `poetry install` por `uv sync`, `poetry build` por `uv build`
- Documentación (`AGENTS.md`, `docs/install.md`, `docs/DEVELOPING.md`): actualizar comandos
- Scripts de shell (`scripts/`): **sin cambios** — ya usan `python -m` directamente
- **Sin cambios** en código de aplicación, tests, o configs de workflow

## Capabilities

### New Capabilities

Ninguna. Este cambio es puramente de infraestructura: no introduce nuevas capacidades ni modifica el comportamiento del paquete en tiempo de ejecución.

### Modified Capabilities

Ninguna. No hay cambios a nivel de especificación funcional.

## Non-goals

- No se actualizan versiones de dependencias (se mantienen los rangos exactos actuales)
- No se refactoriza código de aplicación
- No se cambian configs de Hydra, modelos, o pipelines
- No se modifica el proceso de publicación a PyPI más allá del tooling de build

## Impact

- **Entorno local**: developers necesitan instalar `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **CI**: primer build post-migración tendrá caché fría (nuevo hash de lockfile)
- **Pre-commit**: se pierde validación automática de lockfile; alternativamente se puede añadir hook `uv-lock-check` si el repo `astral-sh/uv-pre-commit` madura
- **Build**: `hatchling` empaqueta correctamente el directorio `gfmrag/`; el kernel CUDA `rspmm` se compila via `torch.utils.cpp_extension` y no depende del build backend
