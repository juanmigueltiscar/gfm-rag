# `gfmrag/graph_index_construction/` — Codemap

## Responsibility

**Graph Index Construction Pipeline** — a composable extraction-and-linking pipeline that transforms raw unstructured document corpora into structured knowledge graphs (stage 1) and then enriches QA datasets with entity-linked supervision signals (stage 2). It is the offline data preparation layer that sits between the raw dataset ingestion boundary and the graph-indexed dataset consumed by the GFM retrieval models.

In standard software engineering terms, this module is a **batch ETL (Extract-Transform-Load) pipeline** with two stages:

1. **Graph ETL**: Document corpus → KG triples → typed node/edge/relation CSV files (`stage1`)
2. **SFT ETL**: QA pairs + KG → entity-linked start/target nodes for supervised fine-tuning (`stage2`)

---

## Design Patterns

### 1. Strategy Pattern — Pluggable Model Abstractions

Three abstract base classes define interface contracts, with concrete implementations swappable at runtime. All are injectable via composition.

| Interface (ABC) | Contract Methods | Concrete Strategies | Purpose |
|---|---|---|---|
| `BaseNERModel` | `__call__(text: str) -> list` | `LLMNERModel` | NER via LLM backends (OpenAI, Together, Ollama, llama.cpp) |
| `BaseOPENIEModel` | `__call__(text: str) -> dict` | `LLMOPENIEModel` | Two-stage OpenIE: NER → triple extraction |
| `BaseELModel` | `index(entity_list)`, `__call__(ner_list, topk) -> dict` | `DPRELModel`, `NVEmbedV2ELModel`, `ColbertELModel` | Entity linking via DPR/bi-encoder or ColBERT late-interaction |

Each base class lives in `<subpackage>/base_model.py` and is re-exported via `<subpackage>/__init__.py`.

### 2. Factory Method — LangChain Client Factory

`init_langchain_model()` in `langchain_util.py` is a static factory that maps the string literal parameter `llm` (`"openai"`, `"nvidia"`, `"together"`, `"ollama"`, `"llama.cpp"`) to the appropriate LangChain `ChatModel` subclass (`ChatOpenAI`, `ChatNVIDIA`, `ChatTogether`, `ChatOllama`, `ChatLlamaCpp`). It isolates all model-provider-specific instantiation logic (API keys, context windows, VRAM flags) in one function.

### 3. Template Method Pattern — SFT Constructor Hierarchy

`BaseSFTConstructor` declares `prepare_data(data_root, data_name, file) -> list[dict]` as the abstract primitive operation. Three concrete subclasses implement the template:

- `GFMRAGConstructor` — NER on question → EL → `start_nodes` (entities) + `target_nodes` (entities from supporting docs)
- `GFMReasonerConstructor` — Same as above but also NER on answer → answer entities in `target_nodes`
- `HippoRAG2Constructor` — Dense embedding + FAISS index + fact retrieval + optional LLM reranking → `start_nodes`/`target_nodes` with entity/doc types

All share the same `tmp_dir` property pattern (generates `<root>/<data_name>/`).

### 4. Composition — Orchestrator Classes

- `KGConstructor` composes `BaseOPENIEModel` + `BaseELModel` to run the full graph-building pipeline.
- `GFMRAGConstructor` / `GFMReasonerConstructor` compose `BaseNERModel` + `BaseELModel`.
- `HippoRAG2Constructor` composes `BaseTextEmbModel` (from sibling package `gfmrag.text_emb_models`) + `DSPyFilter`.

### 5. Data Transfer Objects (TypedDicts)

`base_graph_constructor.py` defines four `TypedDict` types that serve as the canonical data contract for graph data flowing between modules:

```python
class Node(TypedDict):     name, type, attributes, uid(optional)
class Edge(TypedDict):     source, relation, target, attributes
class Relation(TypedDict): name, attributes, uid(optional)
class Graph(TypedDict):    nodes, relations, edges
```

### 6. Parallel Worker Pool

Two orchestrators (`KGConstructor.open_ie_extraction`, `GFMRAGConstructor.prepare_data`) use `multiprocessing.dummy.Pool` (thread pool) with `imap` for concurrent LLM inference. The `HippoRAG2Constructor` uses both sequential and `Pool.map` for fact reranking.

### 7. Prompt Template (One-Shot + System Messages)

`openie_extraction_instructions.py` pre-builds LangChain `ChatPromptTemplate` instances (`ner_prompts`, `openie_post_ner_prompts`) with a system instruction, one-shot human/assistant example pair, and a templated user input slot. These are consumed by `LLMOPENIEModel.ner()` and `.openie_post_ner_extract()`.

### 8. DSPy-Style Reranker

`hipporag2/rerank.py` implements an LLM-based fact reranker (`DSPyFilter`) using a structure inspired by the DSPy framework: a message template with few-shot demos, a typed output schema (`Fact` pydantic model), and a custom parser for the `[[ ## field ## ]]` markup format.

---

## Data & Control Flow

### Stage 1 — Knowledge Graph Construction

```
Raw documents.json
    │
    ▼
KGConstructor.build_graph(data_root, data_name)
    │
    ├─ (1) open_ie_extraction(raw_path)
    │       │
    │       ├─ Reads raw/documents.json (dict of title → passage)
    │       ├─ Optionally prepends title to each passage (add_title=True)
    │       ├─ Skips cached openie_results.jsonl if exists
    │       ├─ ThreadPool.imap(self.open_ie_model, remaining_passages)
    │       │       └─ LLMOPENIEModel.__call__(passage)
    │       │              ├─ self.ner(text)              → named entity list
    │       │              └─ self.openie_post_ner_extract → triple list
    │       └─ Appends results to openie_results.jsonl
    │
    ├─ (2) build_kg(openie_result_path)
    │       │
    │       ├─ Parses JSONL → extracts entities, triples, stats
    │       ├─ Normalizes phrases via processing_phrases() (lowercase, strip non-alphanum)
    │       ├─ Builds graph dict {(head, tail): relation}
    │       ├─ Optionally calls augment_graph() via el_model for "equivalent" edges
    │       │       ├─ el_model.index(processed_phrases)
    │       │       └─ el_model(processed_phrases, topk) → similarity neighbors
    │       ├─ Saves passage_info.json (per-document entity + triple lists)
    │       └─ Saves kg.txt (DELIMITER-separated head/relation/tail lines)
    │
    └─ (3) Builds return Graph dict
            ├─ Reads passage_info.json → document2entities mapping
            ├─ Creates "document" type nodes from raw/documents.json
            ├─ Creates "entity" type nodes from kg.txt triples
            ├─ Creates "is_mentioned_in" edges (entity → document)
            └─ Returns Graph {nodes, relations, edges}
```

### Stage 2 — SFT Data Preparation

```
Graph CSV files (nodes.csv, edges.csv, relations.csv)
    │
    ▼
GFMRAGConstructor.prepare_data(data_root, data_name, file)
 SFTConstructor.prepare_data(...)
    │
    ├─ Reads nodes.csv → entity names filtered by type=="entity"
    ├─ Reads edges.csv → document2entities from "is_mentioned_in" edges
    ├─ NER on question (LLMNERModel.__call__)
    │       └─ Cached in tmp_dir/ner_results.jsonl
    ├─ EL: el_model.index(entities) → el_model(ner_ents, topk=1)
    └─ Assembles final samples with start_nodes / target_nodes

GFMReasonerConstructor (variant):
    ├─ NER on BOTH question AND answer
    └─ target_nodes.entity comes from answer entities (not supporting documents)

HippoRAG2Constructor (variant):
    ├─ FAISS index over node text embeddings (per type: entity, document, etc.)
    ├─ FAISS index over fact triples (edges minus "is_mentioned_in")
    ├─ Retrieves fact candidates per query via dense search
    ├─ Optionally reranks facts via DSPyFilter.llm_call()
    ├─ graph_search_with_fact_entities() aggregates fact-linked entity scores
    ├─ Optional dense passage retrieval fallback when no facts remain
    └─ Assembles start_nodes (entity/doc types) + target_nodes (entity + doc)
```

### Consumer Orchestration

```python
# gfmrag/graph_indexer.GraphIndexer.index_data(dataset_cfg)
graph = graph_constructor.build_graph(root, data_name)        # → stage1 CSVs
train_data = sft_constructor.prepare_data(root, data_name, "train.json")  # → stage1 JSON
test_data  = sft_constructor.prepare_data(root, data_name, "test.json")
```

```python
# gfmrag/gfmrag_retriever.GFMRetriever.from_index(...)
# If stage1/ missing:
graph_constructor.build_graph(data_dir, data_name)         # → stage1 CSVs
# At inference time:
ner_model(query)                    # NER
el_model(mentioned_entities, topk)  # EL → node mask
text_emb_model.encode([query])      # query embedding
graph_retriever(graph, input)       # GNN forward pass
```

### State Transitions

1. **Cache-first execution**: All pipeline steps check for existing cached outputs (e.g., `openie_results.jsonl`, `ner_results.jsonl`, cached entity embeddings) before recomputing. This is controlled by `force: bool` flags.
2. **Embedding caching**: `DPRELModel` stores entity embeddings as `.pt` files keyed by MD5 fingerprint. `ColbertELModel` uses `PLAID` indexes persisted to disk with metadata fingerprints.
3. **Stage dependency**: Stage 2 (SFT) requires stage 1 (graph CSVs) to exist. Raises `FileNotFoundError` with explicit message if missing.
4. **Lazy directory creation**: `tmp_dir` properties create directories on first access using `os.makedirs`.

---

## Integration Points

### Public API Surface (re-exported via `__init__.py` files)

| Subpackage | Exported Symbols |
|---|---|
| `graph_index_construction.graph_constructors` | `BaseGraphConstructor`, `KGConstructor` |
| `graph_index_construction.sft_constructors` | `BaseSFTConstructor`, `GFMRAGConstructor`, `GFMReasonerConstructor`, `HippoRAG2Constructor` |
| `graph_index_construction.entity_linking_model` | `BaseELModel`, `ColbertELModel`, `DPRELModel`, `NVEmbedV2ELModel` |
| `graph_index_construction.ner_model` | `BaseNERModel`, `LLMNERModel` |
| `graph_index_construction.openie_model` | `BaseOPENIEModel`, `LLMOPENIEModel` |

### Consumers (callers within the same package)

- **`gfmrag.graph_indexer.GraphIndexer`** (via `gfmrag/graph_indexer.py`)
  - Calls `BaseGraphConstructor.build_graph()` → writes to `processed/stage1/`
  - Calls `BaseSFTConstructor.prepare_data()` → writes `train.json` / `test.json`
  - This is the primary batch entrypoint, invoked by the `python -m gfmrag.workflow.*` scripts.

- **`gfmrag.gfmrag_retriever.GFMRetriever`** (via `gfmrag/gfmrag_retriever.py`)
  - Imports `BaseELModel`, `BaseGraphConstructor`, `BaseNERModel` at module level
  - `from_index()` calls `BaseGraphConstructor.build_graph()` on-demand if stage1 is missing
  - `prepare_input_for_graph_retriever()` calls `BaseNERModel.__call__()` + `BaseELModel.__call__()` at inference time

### External Package Dependencies

| Dependency | Used By | Purpose |
|---|---|---|
| `langchain-openai`, `langchain-community`, `langchain-nvidia-ai-endpoints`, `langchain-together` | `langchain_util.py`, `ner_model/llm_ner_model.py`, `openie_model/llm_openie_model.py` | LLM client instantiation for NER and OpenIE |
| `sentence-transformers` | `entity_linking_model/dpr_el_model.py` | Dense entity embeddings (DPR-style) |
| `pylate` | `entity_linking_model/colbert_el_model.py` | ColBERT late-interaction encoding + PLAID indexing |
| `faiss` | `sft_constructors/hipporag2_constructor.py` | Dense node/fact similarity search |
| `pandas` | `graph_constructors/kg_constructor.py`, `sft_constructors/*.py` | CSV read/write for graph files |
| `numpy` | Multiple files | Array operations, normalization |
| `tqdm` | Multiple files | Progress bars for long-running loops |
| `omegaconf` | `gfmrag.graph_indexer` (not in this dir but drives it) | Hydra configuration for dataset configs |

### Internal Cross-Module Dependencies (within this directory)

| Source | Import Target | Why |
|---|---|---|
| `graph_constructors/kg_constructor.py` | `entity_linking_model.BaseELModel` | Entity similarity during graph augmentation |
| `graph_constructors/kg_constructor.py` | `openie_model.BaseOPENIEModel` | OpenIE triple extraction |
| `sft_constructors/gfm_rag_constructor.py` | `entity_linking_model.BaseELModel` + `ner_model.BaseNERModel` | NER + EL for SFT data |
| `sft_constructors/gfm_reasoner_constructor.py` | Same as above | Same pattern |
| `sft_constructors/hipporag2_constructor.py` | `text_emb_models.BaseTextEmbModel` (sibling package) | Text embedding for dense retrieval |
| `openie_model/llm_openie_model.py` | `langchain_util.init_langchain_model` | LLM client creation |
| `openie_model/llm_openie_model.py` | `openie_extraction_instructions.ner_prompts`, `.openie_post_ner_prompts` | Prompt template constants |
| `ner_model/llm_ner_model.py` | `langchain_util.init_langchain_model` | LLM client creation |
| `entity_linking_model/*.py` | `utils.processing_phrases` | Text normalization before encoding |

### File System Contract (with external consumers)

| Path | Format | Produced By | Consumed By |
|---|---|---|---|
| `<root>/<data_name>/raw/documents.json` | `dict[str, str]` | User/data loader | `KGConstructor.open_ie_extraction()` |
| `<root>/<data_name>/processed/stage1/nodes.csv` | CSV | `GraphIndexer.index_data()` | `GFMRetriever.from_index()`, SFT constructors, `GraphIndexDataset` |
| `<root>/<data_name>/processed/stage1/edges.csv` | CSV | Same | Same |
| `<root>/<data_name>/processed/stage1/relations.csv` | CSV | Same | Same |
| `<root>/<data_name>/processed/stage1/train.json` | JSON | `GraphIndexer.index_data()` | `GraphIndexDataset` stage2 processing |
| `<root>/<data_name>/processed/stage1/test.json` | JSON | Same | Same |
| `tmp/kg_construction/<data_name>/` | Various | `KGConstructor` | Internal caching |
| `tmp/qa_construction/<data_name>/` | Various | SFT constructors | Internal caching |
