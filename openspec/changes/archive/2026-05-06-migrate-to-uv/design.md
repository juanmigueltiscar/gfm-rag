## Context

Actualmente el proyecto usa Poetry para gestión de dependencias, lockfile, y build. El build backend es `poetry.core.masonry.api`. Hay 3 workflows de CI, un hook de pre-commit (`poetry-check`), y documentación que referencia comandos `poetry`.

La migración a uv es puramente de infraestructura: no cambia funcionalidad del paquete, no toca código de aplicación, no modifica tests ni configs de modelo.

## Goals / Non-Goals

**Goals:**
- `pyproject.toml` compatible con uv en formato PEP 621
- Lockfile `uv.lock` generado y sincronizado
- CI/CD funcionando con uv en lugar de poetry
- Pre-commit sin hooks rotos
- Documentación actualizada (AGENTS.md, docs/)

**Non-Goals:**
- Cambiar rangos de versiones de dependencias
- Refactorizar código de aplicación, tests, o configs de workflow
- Modificar el proceso de publicación a PyPI (solo el tooling)

## Decisions

### 1. Build backend: hatchling

**Opción elegida**: `hatchling`

`hatchling` es el build backend moderno más simple para un paquete con un solo directorio (`gfmrag/`). Alternatives consideradas:

| Backend | Veredicto |
|---------|-----------|
| **hatchling** | Elegido. Configuración mínima: `packages = ["gfmrag"]` |
| setuptools | Más boilerplate (`setup.cfg` o `setup.py`) sin beneficio |
| flit-core | Similar a hatchling pero hatchling está más extendido |
| poetry-core | Podría mantenerse (uv lo soporta como backend) pero ata el build a poetry |

### 2. Formato de dependencias: PEP 508 (no `^`)

`hatchling` valida metadatos PEP 508 y no soporta `^`. Aunque uv acepta `^` en `[project]` durante resolución, hatchling lo rechaza al construir el wheel. Todas las dependencias se convirtieron a formato PEP 508 (`>=X.Y.Z,<X+1.0.0`).

### 3. Grupos de dependencias: `[dependency-groups]` (PEP 735)

Usamos el formato nativo de uv en lugar de `[project.optional-dependencies]` porque estos grupos (dev, doc) no deben publicarse en PyPI. El comando `uv sync` los instala por defecto; `uv sync --no-group dev --no-group doc` instala solo las principales.

### 4. CI: `astral-sh/setup-uv`

Esta action oficial maneja instalación de uv, caching, y soporta `uv sync`, `uv build`, etc. Reemplaza a `snok/install-poetry`.

### 5. Pre-commit: eliminar `poetry-check`

El hook validaba consistencia entre pyproject.toml y poetry.lock. Con uv perdería sentido. Eliminamos el hook del bloque `python-poetry/poetry`. Si en el futuro `astral-sh/uv-pre-commit` ofrece un reemplazo estable, se puede añadir.

## pyproject.toml — estructura final

```toml
[project]
name = "gfmrag"
version = "2.0.0"
description = "Graph Foundation Model for Retrieval Augmented Generation"
authors = [
    {name = "Linhao Luo", email = "linhao.luo@monash.edu"},
    {name = "Zicheng Zhao", email = "zicheng.zhao@njust.edu.cn"},
]
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch>=2.4.1",
    "torch-geometric^2.4.0",
    "ninja^1.11.1.1",
    "easydict^1.13",
    "pyyaml^6.0.2",
    "tqdm^4.66.5",
    "sentence-transformers>=3.4.1",
    "hydra-core^1.3.2",
    "python-dotenv^1.0.1",
    "wandb^0.18.5",
    "transformers^4.52.4",
    "openai^2.29.0",
    "tiktoken^0.8.0",
    "langchain^0.3.9",
    "langchain-openai^0.3.33",
    "langchain-together^0.3.1",
    "langchain-community^0.3.9",
    "langchain-nvidia-ai-endpoints^0.3.9",
    "pylate^1.4.0",
    "faiss-gpu-cu12^1.9.0.post1",
    "vllm^0.10.2",
    "pymetis^2025.2.2",
]

[dependency-groups]
dev = [
    "pre-commit^4.0.1",
    "types-pyyaml^6.0.12.20240917",
    "mypy^1.12.1",
    "pytest^8.3.3",
]
doc = [
    "mkdocs^1.6.1",
    "mkdocstrings^0.27.0",
    "mkdocstrings-python^1.13.0",
    "mkdocs-autorefs^1.3.0",
    "mkdocs-material^9.5.50",
    "mkdocs-same-dir^0.1.3",
    "mike^2.1.3",
]

[tool.hatch.build.targets.wheel]
packages = ["gfmrag"]

# tool.ruff, tool.ruff.lint, mypy.ini se mantienen IDÉNTICOS
# ...

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Cambios en CI/CD

### code-quality.yml
```diff
- uses: snok/install-poetry@v1
+ uses: astral-sh/setup-uv@v3
- run: poetry install --no-interaction --no-root --only dev
+ run: uv sync --no-group doc
- key: venv-${{ hashFiles('**/poetry.lock') }}
+ key: venv-${{ hashFiles('**/uv.lock') }}
```

### python-publish.yml
```diff
- uses: snok/install-poetry@v1
+ uses: astral-sh/setup-uv@v3
- run: poetry install --no-interaction --no-root
+ run: uv sync --no-group dev --no-group doc
- run: echo "version=`poetry version --short`"
+ run: echo "version=`python -c 'import tomllib; f=open(\"pyproject.toml\",\"rb\"); print(tomllib.load(f)[\"project\"][\"version\"])'`"
- run: poetry build
+ run: uv build
```

### gh-pages.yml
```diff
- uses: snok/install-poetry@v1
+ uses: astral-sh/setup-uv@v3
- run: poetry install --no-interaction --no-root --only doc
+ run: uv sync --no-group dev
```

## Comandos uv equivalentes

| Poetry | uv |
|--------|----|
| `poetry install` | `uv sync` |
| `poetry add pkg` | `uv add pkg` |
| `poetry remove pkg` | `uv remove pkg` |
| `poetry run cmd` | `uv run cmd` |
| `poetry build` | `uv build` |
| `poetry publish` | `uv publish` |
| `poetry lock` | `uv lock` |

## Risks / Trade-offs

| Riesgo | Mitigación |
|--------|------------|
| **Resolución diferente**: uv puede generar lockfile con versiones distintas a poetry.lock | Comparar `uv lock` output contra `poetry.lock`; ejecutar tests completos post-migración |
| **Cache frío en CI**: primera ejecución post-migración no tiene cache | Aceptable; ocurre una vez |
| **hatchling empaqueta distinto**: el wheel generado podría incluir/excluir archivos de forma diferente | `hatchling` incluye todo lo que está en `packages = ["gfmrag"]`. Verificar con `uv build` y `tar tzf dist/*.tar.gz` |
| **Devs sin uv**: necesitan instalarlo | Documentar en AGENTS.md; instalación es un comando (`curl`) |
| **poetry.lock huérfano**: queda en el repo si no se borra | Incluir `poetry.lock` en `.gitignore` o borrarlo explícitamente |
