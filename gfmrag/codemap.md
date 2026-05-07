# `gfmrag/` — Package Codemap

## Responsibility

`gfmrag` is the **top-level application package** of a Graph Foundation Model for Retrieval-Augmented Generation (GFM-RAG) system. It provides the public API surface and orchestrates the full pipeline:

1. **Graph Indexing** — batch construction of knowledge graphs from raw document collections.
2. **Retrieval** — query-time end-to-end retrieval over the indexed graph using a GNN-based retriever.
3. **Training** — KGC and SFT training loops for the GNN backbone models.
4. **Evaluation** — multi-dataset QA and retrieval evaluation harnesses.
5. **LLM Prompting** — prompt assembly for downstream LLM reasoning over retrieved context.

The package is structured as a **Hydra-driven application** where every configurable component is wired via OmegaConf/DictConfig at entry points under `workflow/`.

---

## Design Patterns

### 1. Abstract Base Classes (Strategy Pattern via Protocol/ABC)

Every extensible component defines an abstract interface in its own `base_*.py`, with concrete implementations as strategies:

| ABC / Interface | Module | Concrete Implementations |
|---|---|---|
| `BaseGNNModel` | `models/base_model.py` | `ultra.ULTRA`, `ultra.GraphReasoner` (under `models/`) |
| `BaseTextEmbModel` | `text_emb_models/base_model.py` | `NVEmbedV2`, `Qwen3TextEmbModel` |
| `BaseLoss` | `losses.py` | `BCELoss`, `ListCELoss`, `KLDivLoss`, `MSELoss` |
| `BaseGraphConstructor` | `graph_index_construction/graph_constructors/base_graph_constructor.py` | `KGConstructor` |
| `BaseSFTConstructor` | `graph_index_construction/sft_constructors/base_sft_constructor.py` | `GFMRagSFTConstructor`, `GFMReasonerSFTConstructor`, `HippoRAG2SFTConstructor` |
| `BaseNERModel` | `graph_index_construction/ner_model/base_model.py` | `LLMNERModel` |
| `BaseELModel` | `graph_index_construction/entity_linking_model/base_model.py` | `DPRELModel`, `ColBERTELModel` |
| `BaseLanguageModel` | `llms/base_language_model.py` | `ChatGPT`, `HfCausalModel`, `MistralModel`, `GeminiModel` |
| `BaseTrainer` | `trainers/base_trainer.py` | `KGCTrainer`, `SFTTrainer` |
| `BaseEvaluator` | `evaluation/base_evaluator.py` | `HotpotQAEvaluator`, `MusiqueEvaluator`, `RetrievalEvaluator`, `TwoWikiQAEvaluator` |

### 2. Static Factory Method

`GFMRetriever.from_index()` (in `gfmrag_retriever.py`) is a **static factory method** that:

- Detects whether stage1 CSVs already exist.
- Optionally delegates graph building to a `BaseGraphConstructor`.
- Loads a pretrained GNN model + config from a HuggingFace Hub path via `utils.load_model_from_pretrained()`.
- Recovers the dataset class from the checkpoint's `dataset_config.class_name` string (via `hydra.utils.get_class`), achieving **reflective instantiation**.
- Assembles and returns a fully initialized `GFMRetriever`.

### 3. Template Method

`GraphIndexer.index_data()` (in `graph_indexer.py`) defines the **invariant skeleton** of the batch indexing pipeline:

1. Build graph → write nodes.csv, edges.csv, relations.csv.
2. If train.json/test.json exist in raw/, pass through `sft_constructor.prepare_data()` → write to stage1/.

The two substeps are always performed, but the graph structure and SFT format depend on which `BaseGraphConstructor` / `BaseSFTConstructor` strategies are plugged in.

### 4. Config-Driven Composition (Hydra / OmegaConf)

All wiring is externalized into YAML config files under `workflow/config/`. The configs are hierarchical:

```
workflow/config/
├── gfm_rag/          ← GFM-RAG-8M model family
│   ├── kgc_training.yaml
│   ├── sft_training.yaml
│   ├── qa_inference.yaml
│   ├── qa_ircot_inference.yaml
│   ├── index_dataset.yaml
│   └── visualize_path.yaml
├── gfm_reasoner/     ← G-reasoner-34M model family
│   ├── kgc_trianing.yaml
│   ├── sft_training.yaml
│   ├── sft_training_w_answer.yaml
│   ├── qa_inference.yaml
│   ├── stage3_qa_ircot_inference.yaml
│   ├── index_dataset.yaml
│   └── visualize_path.yaml
├── graph_constructor/
├── sft_constructor/
├── ner_model/
├── el_model/
├── text_emb_model/
├── qa_prompt/
├── agent_prompt/
├── doc_ranker/
├── qa_evaluator/
└── wandb/
```

Configs are loaded with `@hydra.main(version_base=None, config_path=…)` on workflow launchers. Components are instantiated via `hydra.utils.instantiate()` (see `gfmrag_retriever.py` line 292).

### 5. Deferred/Lazy Import

`gfmrag_retriever.py` line 93–95 defers `from gfmrag.models.ultra import query_utils` until inside `retrieve()` to **avoid circular import** at module load time. This is an intentional coupling point between `gfmrag_retriever` and `gfmrag.models.ultra`.

---

## Data & Control Flow

### A. Batch Indexing (`GraphIndexer.index_data`)

```
Input:  dataset_cfg (DictConfig with root, data_name)
        raw/documents.json

Step 1: Check if processed/stage1/{nodes,edges,relations}.csv exist.
        If missing or force=True:
            graph = graph_constructor.build_graph(root, data_name)
            Write nodes.csv, edges.csv, relations.csv

Step 2: If raw/train.json exists and processed/stage1/train.json missing:
            train_data = sft_constructor.prepare_data(root, data_name, "train.json")
            Write processed/stage1/train.json
        Same for test.json.

Output: processed/stage1/{nodes,edges,relations}.csv
        processed/stage1/{train,test}.json (optional)
```

### B. End-to-End Retrieval (`GFMRetriever.from_index` → `.retrieve`)

```
Input:  data_dir, data_name, model_path, ner_model, el_model,
        [graph_constructor], [force_reindex]

Phase 1 — Graph construction (static factory):
    1. Check all stage1 CSVs exist via utils.check_all_files_exist().
       If not:
         a. Verify raw/documents.json exists.
         b. graph_constructor.build_graph(data_dir, data_name)
         c. Write stage1 CSVs.
    2. utils.load_model_from_pretrained(model_path) → (graph_retriever, model_config)
    3. _load_qa_data_from_model_config() → GraphIndexDataset
       (reads model_config.dataset_config to resolve the dataset class
        and kwargs, then instantiates with root + data_name)
    4. el_model.index(list(qa_data.node2id.keys()))
    5. Read nodes.csv → nodes_df (set_index on uid/name)
    6. instantiate(text_emb_model_cfgs) → text_emb_model
    7. Assemble GFMRetriever(qa_data, text_emb_model, ner_model, el_model,
                            graph_retriever, node_info, device)

Phase 2 — Online retrieval:
    retriever.retrieve(query, top_k, [target_types]):
    1. prepare_input_for_graph_retriever(query):
        a. ner_model(query) → mentioned entity strings
        b. el_model(entity_strings, topk=1) → linked entity dicts
        c. Filter entities to those in qa_data.node2id
        d. entities_to_mask(entity_ids, num_nodes) → start_nodes_mask [1, N]
        e. text_emb_model.encode([query], is_query=True) → question_embeddings [1, D]
        f. Return {"question_embeddings": ..., "start_nodes_mask": ...}
    2. query_utils.cuda(..., device) → move to GPU
    3. graph_retriever(graph, input) → pred [1, num_nodes]
    4. For each target_type:
        - Slice pred to node_ids of that type
        - torch.topk → top_k scores & indices
        - Map indices back to original node IDs + attributes from node_info
        - Build result dicts
    5. Return dict[target_type → list[dict{id, type, attributes, score}]]

Output: dict[str, list[dict]]  (ranked nodes per target type)
```

### C. Training Loop (via `trainers/`)

```
Launcher (workflow/ scripts) loads config → instantiate:
    - GraphIndexDataset (training data)
    - BaseGNNModel
    - BaseLoss (from losses.py)
    - Optimizer

Trainer.train():
    for epoch:
        for batch in dataloader:
            pred = model(graph, batch)
            loss = loss_fn(pred, batch.target)
            loss.backward()
            optimizer.step()

Two training modes:
    - KGC training (KGCTrainer): link prediction / graph completion
    - SFT training (SFTTrainer): supervised fine-tuning on QA pairs
```

### D. QA Inference (via `workflow/` scripts)

```
Launcher loads config → instantiate:
    - GFMRetriever or equivalent retrieval pipeline
    - QAPromptBuilder
    - BaseLanguageModel (LLM)

For each question:
    1. retrieved = retriever.retrieve(question, top_k)
    2. prompt = prompt_builder.build_input_prompt(question, retrieved, [thoughts])
    3. answer = llm.generate(prompt)
    4. evaluator.score(answer, ground_truth)
```

### Data Format State Machine

```
raw/documents.json
    │
    ▼  [BaseGraphConstructor.build_graph()]
    │
processed/stage1/           ◄── GraphIndexer entry point
├── nodes.csv               ─── columns: [uid|name, type, attributes, ...]
├── edges.csv               ─── columns: [head, tail, relation, ...]
├── relations.csv           ─── columns: [relation_id, name, ...]
├── train.json (optional)   ─── SFT-formatted training data
└── test.json (optional)    ─── SFT-formatted test data
    │
    ▼  [GraphIndexDataset.process()]
    │
processed/stage2/{fingerprint}/
├── graph.pt                ─── torch_geometric Data object
├── node2id.json            ─── str → int mapping
├── rel2id.json             ─── str → int mapping
├── train.pt (optional)     ─── tensorized training targets
└── test.pt (optional)      ─── tensorized test targets
```

---

## Integration Points

### Package-Level Exports (`__init__.py`)

```python
from .gfmrag_retriever import GFMRetriever  # Main retrieval entrypoint
from .graph_indexer import GraphIndexer  # Main indexing entrypoint
from . import trainers  # Training subpackage
```

### Internal Dependencies (Module → Module)

| Consumer | Produced By | Mechanism |
|---|---|---|
| `gfmrag_retriever.py` | `gfmrag.graph_index_datasets` | `GraphIndexDataset` (import + subclass resolution via `get_class`) |
| `gfmrag_retriever.py` | `gfmrag.utils` | `load_model_from_pretrained()`, `get_device()`, `check_all_files_exist()`, `entities_to_mask()` |
| `gfmrag_retriever.py` | `gfmrag.models.base_model` | `BaseGNNModel` (type contract for `graph_retriever`) |
| `gfmrag_retriever.py` | `gfmrag.models.ultra.query_utils` | `query_utils.cuda()` (deferred import) |
| `gfmrag_retriever.py` | `gfmrag.text_emb_models` | `BaseTextEmbModel` (instantiated via Hydra) |
| `gfmrag_retriever.py` | `gfmrag.graph_index_construction.ner_model` | `BaseNERModel` (injected) |
| `gfmrag_retriever.py` | `gfmrag.graph_index_construction.entity_linking_model` | `BaseELModel` (injected, `.index()` called internally) |
| `gfmrag_retriever.py` | `gfmrag.graph_index_construction.graph_constructors` | `BaseGraphConstructor` (optional injection) |
| `graph_indexer.py` | `gfmrag.graph_index_construction.graph_constructors` | `BaseGraphConstructor.build_graph()` |
| `graph_indexer.py` | `gfmrag.graph_index_construction.sft_constructors` | `BaseSFTConstructor.prepare_data()` |
| `graph_indexer.py` | `gfmrag.graph_index_datasets` | `GraphIndexDataset.RAW_GRAPH_NAMES` (file name list) |
| `trainers/` | `gfmrag.losses` | `BaseLoss` subclasses for training objectives |
| `trainers/` | `gfmrag.models.base_model` | `BaseGNNModel` forward pass |
| `trainers/` | `gfmrag.graph_index_datasets` | `GraphDatasetLoader` for batched training data |
| `workflow/*.py` | All of the above | Hydra config composition + `instantiate()` wiring |

### External Dependencies

| Dependency | Version Constraint | Purpose |
|---|---|---|
| `torch` (PyTorch) | ≥2.0 | Tensor ops, GNN execution, GPU management |
| `torch_geometric` | — | Graph data structures (`Data`), message-passing primitives |
| `hydra-core` / `omegaconf` | — | Config-driven composition, CLI entry points |
| `transformers` (HuggingFace) | — | Pretrained model loading, tokenizers |
| `sentence-transformers` | — | Text embedding models |
| `pandas` | — | CSV read/write for stage1 graph files |
| `python-dotenv` | — | `.env` loading for API keys |
| `wandb` | — | Training experiment tracking (optional) |

### API Endpoints (CLI Launchers)

These are `python -m gfmrag.workflow.<name>` entry points, all driven by Hydra configs:

| Launcher Config | Purpose |
|---|---|
| `gfm_rag/index_dataset.yaml` | Batch indexing for GFM-RAG-8M |
| `gfm_reasoner/index_dataset.yaml` | Batch indexing for G-reasoner-34M |
| `gfm_rag/kgc_training.yaml` | KGC training for GFM-RAG-8M |
| `gfm_reasoner/kgc_trianing.yaml` | KGC training for G-reasoner-34M (sic, typo in filename) |
| `gfm_rag/sft_training.yaml` | SFT training for GFM-RAG-8M |
| `gfm_reasoner/sft_training.yaml` | SFT training for G-reasoner-34M |
| `gfm_rag/qa_inference.yaml` | Direct QA inference |
| `gfm_rag/qa_ircot_inference.yaml` | Iterative Retrieval-CoT QA |
| `gfm_reasoner/stage3_qa_ircot_inference.yaml` | Stage3 IRCoT inference for reasoner |

### Environment Variable Hooks

| Variable | Required | Used By |
|---|---|---|
| `OPENAI_API_KEY` | Yes (for LLM calls) | `llms/chatgpt.py` |
| `HF_TOKEN` | Yes (for gated HF models) | `utils/util.py` (model loading) |

### Noteworthy Coupling Points

- **`_STAGE1_GRAPH_NAMES`** — captured at `gfmrag_retriever` module import time (line 23) from `GraphIndexDataset.RAW_GRAPH_NAMES`. This is intentionally snapshotted so that test patching of `GraphIndexDataset` does not affect the file name list used by retriever.
- **`gfmrag.models.ultra`** is excluded from ruff linting (`pyproject.toml`) — it is third-party code derived from the ULTRA repository and should not be reformatted.
- **Inverse relations** are created automatically inside `GraphIndexDataset.process_graph()`, doubling the relation count — consumers of the graph should expect this.

---

## Module Map (16 entries)

```
gfmrag/
├── __init__.py                        # Public API: GFMRetriever, GraphIndexer, trainers
├── gfmrag_retriever.py                # End-to-end retrieval (GFMRetriever class)
├── graph_indexer.py                   # Batch graph + SFT data indexing (GraphIndexer class)
├── losses.py                          # Training loss functions (BCELoss, ListCELoss, KLDivLoss, MSELoss)
├── prompt_builder.py                  # LLM prompt assembly (QAPromptBuilder)
├── evaluation/                        # QA & Retrieval evaluators (HotpotQA, Musique, 2Wiki, etc.)
├── graph_index_construction/          # Graph builders, SFT constructors, NER, EL, OIE
│   ├── graph_constructors/            #   BaseGraphConstructor → KGConstructor
│   ├── sft_constructors/              #   BaseSFTConstructor → GFMRag, GFMReasoner, HippoRAG2
│   ├── ner_model/                     #   BaseNERModel → LLMNERModel
│   ├── entity_linking_model/          #   BaseELModel → DPRELModel, ColBERTELModel
│   └── openie_extraction_instructions.py
├── graph_index_datasets/              # Dataset loaders (GraphIndexDataset, V1, DatasetLoader)
├── llms/                              # Language model wrappers (ChatGPT, HfCausal, Mistral, Gemini)
├── models/                            # GNN model definitions (BaseGNNModel, ultra/)
├── text_emb_models/                   # Text embedding models (NVEmbed, Qwen3, base)
├── trainers/                          # Training loops (KGCTrainer, SFTTrainer, BaseTrainer)
├── utils/                             # Shared utilities (qa_utils, dist_graph, setup, wandb)
└── workflow/                          # Hydra entry points + config YAMLs
    ├── config/                        #   All Hydra config files (hierarchical)
    └── *.py                           #   Launcher scripts (index_dataset, train, inference, eval)
```
