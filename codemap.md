# Repository Atlas: gfm-rag

## Project Responsibility
Graph Foundation Model for Retrieval-Augmented Generation (GFM-RAG). A PyPI-publishable Python package that builds knowledge-graph indexes from raw documents, trains GNN-based retrieval models over those graphs, and serves end-to-end multi-hop QA retrieval pipelines. Two model families are supported: GFM-RAG-8M (retrieval) and G-reasoner-34M (reasoning).

## System Entry Points
- `gfmrag/gfmrag_retriever.py` → `GFMRetriever.from_index()` — end-to-end retrieval
- `gfmrag/graph_indexer.py` → `GraphIndexer.index_data()` — batch dataset indexing
- `gfmrag/workflow/` modules — training / QA / IRCoT scripts launched via `python -m` or `torchrun`
- `scripts/` — shell launchers for all pipeline stages

## Data Pipeline (Three-Stage)

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

## Directory Map (Aggregated)

| Directory | Responsibility Summary | Detailed Map |
|-----------|------------------------|--------------|
| `gfmrag/` | Top-level application layer: orchestrates indexing, retrieval, training, evaluation, and LLM prompting. Main entrypoints `GFMRetriever` and `GraphIndexer`. | [View Map](gfmrag/codemap.md) |
| `gfmrag/evaluation/` | Benchmark-specific evaluator strategies for multi-hop QA and retrieval tasks. Consumes JSONL prediction files, returns aggregated metrics. | [View Map](gfmrag/evaluation/codemap.md) |
| `gfmrag/graph_index_construction/` | Two-stage ETL pipeline: graph construction from documents (NER, OpenIE, entity linking) → SFT data preparation. | [View Map](gfmrag/graph_index_construction/codemap.md) |
| `gfmrag/graph_index_datasets/` | Data access and preprocessing layer for graph-indexed datasets. Single-dataset processing (`GraphIndexDataset`) and multi-dataset orchestration (`GraphDatasetLoader`). | [View Map](gfmrag/graph_index_datasets/codemap.md) |
| `gfmrag/llms/` | Inference adapter layer / Strategy abstraction over heterogeneous LLM backends (ChatGPT, Gemini, Mistral, HuggingFace). | [View Map](gfmrag/llms/codemap.md) |
| `gfmrag/models/` | Neural engine: GNN architectures for graph reasoning and document retrieval. Contains GFM-RAG-v1, G-reasoner, and third-party ULTRA model. | [View Map](gfmrag/models/codemap.md) |
| `gfmrag/text_emb_models/` | Polymorphic text embedding abstraction layer decoupling backends (SentenceTransformer, NV-Embed, Qwen3) from consumers. | [View Map](gfmrag/text_emb_models/codemap.md) |
| `gfmrag/trainers/` | Template Method orchestration layer between datasets, models, and evaluation. KGC pre-training and SFT fine-tuning trainers. | [View Map](gfmrag/trainers/codemap.md) |
| `gfmrag/utils/` | Cross-cutting support library: distributed orchestration, model serialization, metric computation, graph partitioning, experiment tracking. | [View Map](gfmrag/utils/codemap.md) |
| `gfmrag/workflow/` | Orchestration / composition root for the full GFM-RAG pipeline. Hydra-driven config composition for indexing, training, QA inference, and IRCoT. | [View Map](gfmrag/workflow/codemap.md) |
| `scripts/` | Shell launch-pad for all pipeline stages across both model families and benchmark evaluation. | [View Map](scripts/codemap.md) |

## Key Design Patterns
- **Strategy** — 10+ ABC hierarchies with concrete implementations (evaluators, LLMs, text embeddings, graph constructors, rankers)
- **Template Method** — `BaseEvaluator.evaluate()`, `BaseTrainer.train()`, `BaseGraphConstructor.construct()`
- **Static Factory Method** — `GFMRetriever.from_index()`, `GraphIndexer.index_data()`
- **Config-Driven Composition** — Hydra/OmegaConf YAML configs with defaults composition
- **Fingerprint-based Caching** — MD5 content-addressable caching for stage2 outputs

## Gotchas
- **`GraphIndexDataset.raw_dir` points to `stage1`**, NOT `raw/` — the naming is misleading.
- Inverse relations are auto-created (doubling relation count). Handled inside `GraphIndexDataset.process_graph()`.
- `gfmrag/models/ultra/` is **excluded from ruff linting** (third-party ULTRA model). Do not reformat it.
- `_STAGE1_GRAPH_NAMES` is captured at module import time in `gfmrag.gfmrag_retriever` so tests can patch `GraphIndexDataset` without side effects.
- Requires env vars: `OPENAI_API_KEY` (for LLM calls), `HF_TOKEN` (for gated HuggingFace models). Copy `.env.example`.
- CUDA toolkit is a **build-time** dependency (compiles `rspmm` extension).