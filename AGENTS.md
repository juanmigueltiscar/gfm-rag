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

**Before every commit**: run `source .venv/bin/activate && pre-commit run --all-files` and ensure all checks pass. The CI (`code-quality` workflow) runs the same command — committing without passing it will break CI. `ruff-format` auto-fixes files in place; re-stage them and commit again if it modifies anything.

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

## Fine-tuning: config.json flags after training

When a model is fine-tuned with GFM-RAG (e.g. via `gfmrag/workflow/`), the `config.json` saved inside the checkpoint directory contains a `dataset_config` section with these flags:

```json
"dataset_config": {
    "use_node_feat": null,
    "use_relation_feat": null,
    "use_edge_feat": null,
    "inverse_relation_feat": null
}
```

**These default to `null` (False) and must be manually corrected after training.** The flags control what is embedded and stored in `stage2/graph.pt`:

| Flag | Controls | Required by |
|---|---|---|
| `use_relation_feat` | `rel_attr` (relation embeddings) in graph.pt | `GraphReasoner` base pass |
| `use_node_feat` | `graph.x` (entity embeddings) in graph.pt | `use_ent_emb: "early-late-fusion"` |

If the production model uses `"use_ent_emb": "early-late-fusion"` (as G-reasoner-34M does), **both must be `true`**:

```json
"use_node_feat": true,
"use_relation_feat": true
```

Setting them to `null` causes `graph.x = None` / `rel_attr = None` in graph.pt, which results in runtime errors:
- `'GlobalStorage' object has no attribute 'rel_attr'` — `use_relation_feat: null`
- `linear(): argument 'input' (position 1) must be Tensor, not NoneType` — `use_node_feat: null`

**After editing `config.json`, delete `stage2/` to force regeneration** — the subdirectory name is an MD5 fingerprint that includes these flags, so a stale stage2 with wrong embeddings will be used otherwise.

> The public `rmanluo/G-reasoner-34M` checkpoint has these flags set correctly. A freshly fine-tuned checkpoint does not — always verify after training.

## Known issues

### Blackwell GPU non-reproducibility (fixed)

On NVIDIA GB10/Blackwell (sm_120, CUDA 13.0, PyTorch 2.11.0), consecutive `retrieve()` calls returned alternating BAD/GOOD results.

**Root cause**: `relation_representations` had 3960 entries (only direct relations from `rel_attr`) but `edge_type` values reach 7919 (direct + inverse). The rspmm CUDA kernel performed an out-of-bounds read on the `relation` tensor for 40767 of 81534 edges. On Blackwell this manifested as alternating output; on other GPUs it worked by accident (adjacent mapped memory).

**Fix** (commit `e62c90e`):
1. Double `relation_representations` with `torch.cat([rel, rel], dim=1)` so the kernel can index `relation[edge_type]` for inverse edges
2. Replace `.expand()` with `.repeat()` everywhere to break view chains
3. Add `detach().clone()` in `QueryNBFNet.forward` as CPU safety net
4. Initialize rspmm output tensor with `at::zeros()` instead of `at::empty()`

### NER non-determinism (unresolved)

End-to-end retrieval varies between runs (~40% of top-5 documents can differ, especially at positions #3-#5) even after the Blackwell fix. This is caused by **NER non-determinism**, not the GNN retrieval (which is now deterministic).

**Root cause**: The NER model (`LLMNERModel` in `llm_ner_model.py`) calls an LLM through **LiteLLM** (`ChatLiteLLM`), which proxies requests to a vLLM server. LiteLLM's `seed` parameter is **not forwarded** to the underlying LLM — it's silently ignored. This means:
- Each NER call produces slightly different entity extractions
- `num_runs` (default 2) mitigates this by taking the union of multiple runs, but doesn't eliminate it
- The `seed` parameter was removed from `LLMNERModel.__init__` in commit `e62c90e` since it was non-functional

**To fix properly**: Bypass LiteLLM and call vLLM directly via its OpenAI-compatible API for NER, or increase `num_runs`. The current `num_runs=2` provides a reasonable compromise — the top-1 result is always the same document (though with different scores), and the overall document set is semantically coherent regardless of variation.

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
