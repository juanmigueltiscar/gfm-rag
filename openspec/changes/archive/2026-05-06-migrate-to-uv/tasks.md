## 1. pyproject.toml

- [x] 1.1 Reescribir `[tool.poetry]` y `[tool.poetry.dependencies]` como `[project]` con PEP 621
- [x] 1.2 Convertir `[tool.poetry.group.dev.dependencies]` y `[tool.poetry.group.doc.dependencies]` a `[dependency-groups]` (PEP 735)
- [x] 1.3 Cambiar build system de `poetry-core` a `hatchling`, añadiendo `[tool.hatch.build.targets.wheel]` con `packages = ["gfmrag"]`
- [x] 1.4 Eliminar `poetry.lock` y generar `uv.lock` con `uv lock`

## 2. Pre-commit

- [x] 2.1 Eliminar hook `poetry-check` del `.pre-commit-config.yaml`

## 3. CI/CD — GitHub Actions

- [x] 3.1 Actualizar `code-quality.yml`: `snok/install-poetry` → `astral-sh/setup-uv`, `poetry install` → `uv sync --no-group doc`, cambiar cache key a `uv.lock`
- [x] 3.2 Actualizar `python-publish.yml`: mismo patrón + reemplazar `poetry version` por lectura directa de pyproject.toml + `poetry build` → `uv build`
- [x] 3.3 Actualizar `gh-pages.yml`: mismo patrón + `poetry install --only doc` → `uv sync --no-group dev`

## 4. Documentación

- [x] 4.1 Actualizar `AGENTS.md`: `poetry install` → `uv sync`, `poetry run` → `uv run`
- [x] 4.2 Actualizar `docs/install.md`: reemplazar `poetry install` con `uv sync`, actualizar tabla de requisitos
- [x] 4.3 Actualizar `docs/DEVELOPING.md`: reemplazar Poetry por uv en tabla de requisitos, comandos de setup y build

