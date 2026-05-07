# AGENTS.md

## Setup & dev commands

```bash
# Install (requires CUDA 12+, Python 3.12; torch >=2.9.0 for CUDA on aarch64)
uv sync
uv sync --group tests   # installs test dependencies (pandas, datasets, einops)
uv sync --group vllm    # optional: installs vllm for Qwen3TextEmbModel
pre-commit install

# Run all quality checks (equiv to CI)
pre-commit run --all-files --show-diff-on-failure

# Run a single test
uv run python -m pytest tests/test_gfmrag_retriever.py -k test_retrieve_top_k
```

**Order matters**: lint/format/mypy all run via pre-commit. CI does NOT run unit tests — tests are not in automation, must be run manually.

## Architecture

- **Single package** `gfmrag` published to PyPI via uv.
- **Two model families** with separate Hydra config trees under `gfmrag/workflow/config/`:
  - `gfm_rag/` — GFM-RAG-8M model
  - `gfm_reasoner/` — G-reasoner-34M model
- **Main entrypoints**:
  - `GFMRetriever.from_index()` — end-to-end retrieval (`gfmrag.gfmrag_retriever`)
  - `GraphIndexer.index_data()` — batch dataset indexing (`gfmrag.graph_indexer`)
  - `gfmrag.workflow.*` modules — training / QA / IRCoT scripts launched via `python -m` or `torchrun`
- **Batch retrieval**: `GFMRetriever.retrieve(query)` accepts `str | list[str]`. Pass a list to run NER in parallel and forward the GNN in a single batched pass. `max_batch_size` (default 4) controls GPU chunk size — set at `__init__`/`from_index()` or override per call.
- **Entity linking models**: `DPRELModel`, `NVEmbedV2ELModel`, `ColbertELModel`, and `VLLMELModel` (remote vLLM server via OpenAI-compatible `/v1/embeddings`). Configure via Hydra `el_model/vllm_el_model.yaml`.

## Data pipeline (three-stage)

```
raw/documents.json  -->  processed/stage1/  -->  processed/stage2/{fingerprint}/
                          nodes.csv              graph.pt
                          relations.csv          node2id.json
                          edges.csv              rel2id.json
                          train.json (opt)       train.pt (opt)
                          test.json (opt)        test.pt (opt)
```

- `stage1` holds the raw graph CSVs. `stage2` holds processed torch files.
- If `stage1` CSVs already exist, `GFMRetriever.from_index()` skips building; pass `graph_constructor` only for fresh builds.
- `stage2` output is **gitignored** (`data/*/processed/stage2/`).

## Gotchas

- **`GraphIndexDataset.raw_dir` points to `stage1`**, NOT `raw/` — the naming is misleading.
- Inverse relations are auto-created (doubling relation count). This is handled inside `GraphIndexDataset.process_graph()`.
- `gfmrag/models/ultra/` is **excluded from ruff linting** (third-party ULTRA model). Do not reformat it.
- `_STAGE1_GRAPH_NAMES` is captured at module import time in `gfmrag.gfmrag_retriever` so tests can patch `GraphIndexDataset` without side effects.
- Requires env vars: `OPENAI_API_KEY` (for LLM calls), `HF_TOKEN` (for gated HuggingFace models). Copy `.env.example`.
- `python-dotenv` is a dependency — `.env` is gitignored.
- CUDA toolkit is a **build-time** dependency (compiles `rspmm` extension).
- `vllm` is **optional** — only needed for `Qwen3TextEmbModel`. Install with `uv sync --group vllm`. Importing `Qwen3TextEmbModel` without vllm raises a descriptive `ImportError`.
- `faiss` has been **removed** — vector search uses `torch` ops directly (`IndexFlatIP` equivalent). `torch >=2.9.0` is required (enables CUDA on aarch64).

## Repository Map

A full codemap is available at `codemap.md` in the project root.

Before working on any task, read `codemap.md` to understand:
- Project architecture and entry points
- Directory responsibilities and design patterns
- Data flow and integration points between modules

For deep work on a specific folder, also read that folder's `codemap.md`.
