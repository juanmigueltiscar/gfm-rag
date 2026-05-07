## ADDED Requirements

### Requirement: vllm is an optional dependency
The `gfmrag` package SHALL NOT require the `vllm` Python package as a mandatory dependency. The `vllm` package SHALL be declared in an optional dependency group (`[project.optional-dependencies].vllm`) in `pyproject.toml`, installable via `pip install gfmrag[vllm]` or `uv sync --group vllm`.

#### Scenario: Install without vllm
- **WHEN** a user installs `gfmrag` with `pip install gfmrag` or `uv sync`
- **THEN** the `vllm` package is NOT installed and the rest of the project functions normally using non-vllm embedding models

#### Scenario: Install with vllm optional group
- **WHEN** a user installs `gfmrag` with `pip install gfmrag[vllm]` or `uv sync --group vllm`
- **THEN** the `vllm>=0.11.0,<0.12.0` package is installed and `Qwen3TextEmbModel` is fully functional

### Requirement: Graceful degradation when vllm is not installed
When the `vllm` package is not installed, importing `gfmrag.text_emb_models` SHALL succeed without raising an `ImportError`. The `Qwen3TextEmbModel` symbol in the package namespace SHALL be `None` when vllm is absent.

#### Scenario: Import text_emb_models without vllm
- **WHEN** `from gfmrag.text_emb_models import BaseTextEmbModel, NVEmbedV2` is executed without vllm installed
- **THEN** the import succeeds and both classes are usable

#### Scenario: Qwen3TextEmbModel is None without vllm
- **WHEN** `from gfmrag.text_emb_models import Qwen3TextEmbModel` is executed without vllm installed
- **THEN** `Qwen3TextEmbModel` is `None`

### Requirement: Descriptive error when instantiating Qwen3TextEmbModel without vllm
Attempting to import `Qwen3TextEmbModel` from its module (`gfmrag.text_emb_models.qwen3_model`) when vllm is not installed SHALL raise an `ImportError` with a message indicating that vllm must be installed and how to install it.

#### Scenario: Direct import of qwen3_model without vllm
- **WHEN** `from gfmrag.text_emb_models.qwen3_model import Qwen3TextEmbModel` is executed without vllm installed
- **THEN** an `ImportError` is raised with a message containing "vllm is required" and installation instructions

#### Scenario: Hydra instantiation without vllm
- **WHEN** Hydra's `instantiate()` attempts to create a `Qwen3TextEmbModel` via `_target_: gfmrag.text_emb_models.qwen3_model.Qwen3TextEmbModel` without vllm installed
- **THEN** an `ImportError` is raised with a descriptive message, not a cryptic "No module named vllm"

### Requirement: Qwen3TextEmbModel unchanged when vllm is installed
When the `vllm` package IS installed, `Qwen3TextEmbModel` SHALL function identically to its current behavior, including local vLLM server startup and remote API client modes.

#### Scenario: Qwen3TextEmbModel with local server
- **WHEN** `Qwen3TextEmbModel("Qwen/Qwen3-Embedding-0.6B")` is instantiated with vllm installed and no `api_base`
- **THEN** a local vLLM server starts and embeddings can be generated

#### Scenario: Qwen3TextEmbModel with remote API
- **WHEN** `Qwen3TextEmbModel("Qwen/Qwen3-Embedding-0.6B", api_base="http://localhost:8000/v1")` is instantiated with vllm installed
- **THEN** the model connects to the remote vLLM server and embeddings can be generated
