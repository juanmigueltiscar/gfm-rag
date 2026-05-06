# `gfmrag/workflow/` — Codemap

## Responsibility

This directory is the **orchestration layer** for the GFM-RAG pipeline. It provides the **entry-point scripts** that compose model components, datasets, trainers, and inference logic via **declarative configuration**. Each `.py` file is a standalone **CLI launcher** that boots a Hydra-managed config tree, instantiates registered Python classes from YAML, and executes a well-defined phase of the ML lifecycle: **data indexing**, **pre-training (KGC)**, **fine-tuning (SFT)**, **QA inference**, or **interactive multi-step reasoning (IRCoT)**.

It does **not** contain model definitions, dataset logic, or trainer implementations — those live in sibling packages (`gfmrag/models/`, `gfmrag/trainers/`, `gfmrag/graph_index_datasets/`). This directory is the **composition root** of the application.

---

## Design Patterns

### 1. Hydra Configuration Management (Entry-point composition)
- Every `.py` carries a `@hydra.main(config_path=..., config_name=...)` decorator that resolves a **hierarchical YAML config tree** via Hydra's `DefaultsList`.
- `hydra.utils.instantiate()` is the universal factory: it introspects `_target_` keys in config dicts and calls the corresponding Python class constructors with the remaining keys as kwargs.
- The config tree is split into **two model family namespaces** (`config/gfm_rag/` and `config/gfm_reasoner/`), each with its own set of YAML files for the same pipeline stages.
- Shared sub-config directories (`ner_model/`, `el_model/`, `graph_constructor/`, `qa_prompt/`, `qa_evaluator/`, `sft_constructor/`, `agent_prompt/`, `doc_ranker/`, `text_emb_model/`, `openie_model/`, `wandb/`) act as **reusable component registries** referenced via `${...}` interpolation in parent configs.

### 2. Abstract Factory / Strategy (via `hydra.utils.instantiate`)
- All component selection is **strategy-based**. The concrete class instantiated at each integration point is determined by the `_target_` key in YAML, enabling runtime swapping without code changes.
- Examples:
  - NER model: `gfmrag.graph_index_construction.ner_model.LLMNERModel` (swappable per config)
  - EL model: `gfmrag.graph_index_construction.el_model.ColBERTELModel` / `DPRELModel` etc.
  - Graph constructor: `gfmrag.graph_index_construction.graph_constructors.KGConstructor`
  - SFT constructor: `gfmrag.graph_index_construction.sft_constructors.GFMRagSFTConstructor` / `HippoRag2SFTConstructor`
  - LLM for inference: `gfmrag.llms.ChatGPT`
  - Trainer: `gfmrag.trainers.KGCTrainer` / `gfmrag.trainers.SFTTrainer`
  - QA evaluator: `gfmrag.evaluation.hotpotqa.HotpotQAEvaluator` (per-dataset)
  - Prompt builder: `gfmrag.prompt_builder.QAPromptBuilder` (per-dataset per-mode)

### 3. Pipeline Stage / Template Method
- Each script implements a single stage with a fixed **sequence of phases**:
  1. **Init distributed** (`utils.init_distributed_mode`)
  2. **Init datasets** (`utils.init_multi_dataset` or `GraphDatasetLoader`)
  3. **Init model** (`instantiate` or `utils.load_model_from_pretrained`)
  4. **Init trainer** (plugs model + datasets + optimizer + loss functions)
  5. **Execute** (`trainer.train()` / `trainer.predict()`)
  6. **Save & Cleanup** (`utils.save_model_to_pretrained`, `utils.cleanup`, `finish_wandb`)
- The scripts are **pure orchestrators** — they never define business logic, only compose and invoke.

### 4. Multi-process Coordination (Distributed Training)
- Both `kgc_training.py` and `sft_training.py` use `torch.distributed` with a **rank-0-coordination** pattern:
  - Output directory broadcast from rank 0.
  - Dataset pre-processing sharded across ranks via `utils.init_multi_dataset(cfg, world_size, rank)`.
  - Wandb initialized and finalized only on rank 0.
  - `utils.synchronize()` barrier before cleanup.

### 5. Fingerprint-based Caching (Indexing)
- `index_dataset.py` uses a **fingerprint** (MD5 of resolved config, minus `force` keys) as a cache key for intermediate graph construction artifacts, avoiding redundant recomputation.

### 6. Iterative Reasoning Loop (IRCoT)
- `qa_ircot_inference.py` implements a **ReAct-style chain-of-thought with retrieval**:
  - `agent_reasoning()` alternates between `GFMRetriever.retrieve()` and `LLM.generate_sentence()`.
  - At each step, retrieved docs are **merged** with previous steps (dedup by ID, max-score wins, capped at `top_k`).
  - The loop terminates when the LLM emits `"So the answer is:"` or `max_steps` is reached.
  - A separate QA prompt then generates the final answer from the accumulated evidence.

### 7. Thread-pooled Batch Inference
- `qa.py` uses `multiprocessing.dummy.Pool` (thread pool) for parallel LLM queries, with `pool.imap` streaming results directly to `prediction.jsonl`.

---

## Data & Control Flow

### Stage 1: Data Indexing (`index_dataset.py`)

```
Raw documents (JSON)
    │
    ▼
index_dataset.py
    ├── get_tmp_dir(cfg.graph_constructor) → fingerprint hash directory
    ├── get_tmp_dir(cfg.sft_constructor)   → fingerprint hash directory
    ├── instantiate(graph_constructor, root=...) → KGConstructor
    ├── instantiate(sft_constructor, root=...)   → SFTConstructor
    ├── GraphIndexer(graph_constructor, sft_constructor)
    └── kg_indexer.index_data(cfg.dataset)
         │
         ├── NER (LLMNERModel) → entities
         ├── OpenIE → relations/triples
         ├── Entity Linking (EL model) → disambiguated IDs
         ├── KG Constructor → stage1 CSVs (nodes.csv, relations.csv, edges.csv)
         ├── SFT Constructor → stage1 training JSON (optional)
         └── GraphIndexDataset.process_graph() → stage2 torch files
              (graph.pt, node2id.json, rel2id.json, train.pt, test.pt)
```

**State transitions:**
- `raw/documents.json` → `processed/stage1/{nodes,relations,edges}.csv` → `processed/stage2/{fingerprint}/{graph.pt,...}`
- `force: True` in config bypasses the cache check.

### Stage 2a: Knowledge Graph Contrastive Pre-training (`kgc_training.py`)

```
stage2 torch files
    │
    ▼
kgc_training.py
    ├── utils.init_distributed_mode(cfg.timeout)
    ├── for each train dataset:
    │     GraphIndexDataset.process() → GraphData (if not cached)
    ├── GraphDatasetLoader(cfg.datasets, train_names) → sharded streaming loader
    ├── model = QueryGNN(QueryNBFNet backbone), via instantiate()
    ├── optimizer = AdamW
    ├── trainer = KGCTrainer(model, loader, optimizer, ...)
    ├── trainer.train()
    │     └── For each batch: negative sampling → score → contrastive loss → backward
    └── utils.save_model_to_pretrained(model, cfg, output_dir)
```

**Metrics tracked:** MR, MRR, Hits@K (K=1,3,10). Logged to wandb.

### Stage 2b: Supervised Fine-Tuning (`sft_training.py`)

```
stage2 torch files + pre-trained KGC model
    │
    ▼
sft_training.py
    ├── utils.init_distributed_mode(cfg.timeout)
    ├── utils.init_multi_dataset(cfg) → feat_dim validation
    ├── If cfg.load_model_from_pretrained:
    │     model = utils.load_model_from_pretrained(path)
    │   Else:
    │     model = instantiate(cfg.model, feat_dim=...)
    ├── GraphDatasetLoader(cfg.datasets, train_names, ...) → training loader
    ├── GraphDatasetLoader(cfg.datasets, valid_names, ...) → eval loader
    ├── optimizer = AdamW
    ├── Loss composition: List[SFTLoss] each with name, weight, target_node_type
    │     └── e.g. [BCELoss(weight=0.3), ListCELoss(weight=0.7)]
    ├── trainer = SFTTrainer(model, optimizer, loss_fns, loaders, ...)
    ├── trainer.train()
    │     └── For each batch: GNN forward → per-loss weighted sum → backward
    ├── If cfg.save_pretrained:
    │     utils.save_model_to_pretrained(model, cfg, output_dir)
    └── If trainer.args.do_predict:
          trainer.predict() → JSON files per dataset
```

**Loss composition pattern:** Multiple loss functions applied to different target node types with configurable weights, supporting standard supervised loss and distillation loss (e.g., MSE for teacher-student).

### Stage 3a: Single-step QA Inference (`qa.py`)

```
Pre-computed retrieval results (JSON) + node CSV
    │
    ▼
qa.py
    ├── Load retrieval_result from cfg.test.retrieved_result_path
    ├── Load & validate nodes CSV (uid/name index, attributes parsed)
    ├── For each sample (thread pool):
    │     ├── Look up node names from uid indices
    │     ├── Build prompt via QAPromptBuilder
    │     ├── llm.generate_sentence(prompt)
    │     └── Write {"id", "question", "answer", "response", "retrieved_result"} to prediction.jsonl
    └── instantiate(qa_evaluator, prediction_file=...).evaluate()
```

**Note:** This stage operates on **already-retrieved** results — it is a pure LLM answer generation step, not an end-to-end retrieval pipeline.

### Stage 3b: Interleaved Retrieval + Reasoning (IRCoT, `qa_ircot_inference.py`)

```
Raw dataset + GFM-RAG pre-trained model
    │
    ▼
qa_ircot_inference.py
    ├── instantiate(ner_model), instantiate(el_model)
    ├── instantiate(graph_constructor) or None (if stage1 already exists)
    ├── GFMRetriever.from_index(data_dir, data_name, model_path, ner, el, graph_constr)
    │     └── Loads/constructs graph → loads GNN model → initializes QA dataset
    ├── instantiate(llm) → e.g. ChatGPT
    ├── QAPromptBuilder(agent_prompt)  # for reasoning steps
    ├── QAPromptBuilder(qa_prompt)    # for final answer
    ├── For each test sample:
    │     ├── agent_reasoning(cfg, retriever, llm, agent_prompt_builder, query)
    │     │     ├── retrieve(query, top_k, target_types) → ranked docs per type
    │     │     ├── build_input_prompt(query, docs, thoughts)
    │     │     ├── llm.generate_sentence(prompt) → thought
    │     │     ├── if "So the answer is:" in thought → break
    │     │     └── retrieve(thought, ...) → merge with existing docs (dedup, max-score)
    │     ├── build_input_prompt(query, retrieved_docs) → final QA prompt
    │     ├── llm.generate_sentence(prompt) → final answer
    │     └── Write result to prediction.jsonl
    ├── instantiate(qa_evaluator, ...).evaluate()
    └── RetrievalEvaluator(prediction_file).evaluate()
```

**Resume capability:** Supports `cfg.test.resume` path — loads previous `prediction.jsonl` and skips already-processed sample IDs.

---

## Integration Points

### Internal Consumers (within `gfmrag` package)

| Consumer Module | Integration | Direction |
|---|---|---|
| `gfmrag.GraphIndexer` | `index_dataset.py` calls `kg_indexer.index_data()` | Static method call |
| `gfmrag.GFMRetriever` | `qa_ircot_inference.py` calls `GFMRetriever.from_index()` | Class factory method |
| `gfmrag.graph_index_datasets.GraphDatasetLoader` | Training scripts use as streaming loader | Constructor injection |
| `gfmrag.graph_index_datasets.GraphIndexDataset` (variants) | Referenced via `cfg.datasets._target_` | Hydra instantiate |
| `gfmrag.models.gfm_rag_v1.QueryGNN` | KGC pre-training model | Hydra instantiate |
| `gfmrag.models.gfm_rag_v1.GNNRetriever` | SFT training model | Hydra instantiate |
| `gfmrag.models.gfm_reasoner.GraphReasoner` | G-reasoner model family | Hydra instantiate |
| `gfmrag.models.ultra.models.QueryNBFNet` | GNN backbone (shared across model families) | Hydra instantiate |
| `gfmrag.trainers.KGCTrainer` | KGC training loop | Hydra instantiate |
| `gfmrag.trainers.SFTTrainer` | SFT training loop | Hydra instantiate |
| `gfmrag.trainers.TrainingArguments` | Shared training hyperparameter object | Hydra instantiate |
| `gfmrag.llms.BaseLanguageModel` (via ChatGPT etc.) | LLM for QA / IRCoT inference | Hydra instantiate |
| `gfmrag.prompt_builder.QAPromptBuilder` | Prompt formatting for LLM | Constructor call |
| `gfmrag.evaluation.RetrievalEvaluator` | Retrieval metric computation | Constructor call |
| `gfmrag.evaluation.*.Evaluator` (per-dataset) | QA metric computation | Hydra instantiate |
| `gfmrag.graph_index_construction.*` | NER, EL, OpenIE, KG construction, SFT construction | Hydra instantiate |
| `gfmrag.losses.*` (BCELoss, ListCELoss, MSELoss) | Composite loss functions | Hydra instantiate |
| `gfmrag.utils.wandb_utils` | Experiment logging | Direct import |
| `gfmrag.utils` | Distributed utils, model save/load, dataset init | Direct import |

### External Dependencies

| Dependency | Role | Used By |
|---|---|---|
| `hydra` (core) | Config resolution, output dir management | All entry points |
| `hydra.utils.instantiate` | Object factory from `_target_` | All entry points |
| `omegaconf.DictConfig` | Typed config access | All entry points |
| `torch` | Tensor ops, distributed, data loading | Training scripts |
| `torch.distributed` | Multi-GPU synchronization | `kgc_training.py`, `sft_training.py` |
| `dotenv` | Environment variable loading | `index_dataset.py` |
| `pandas` | Node CSV loading & validation | `qa.py` |
| `tqdm` | Progress bars | `qa.py`, `qa_ircot_inference.py` |
| `wandb` via `wandb_utils` | Experiment tracking | Training scripts |

### Config Hooks (Hydra Defaults List Composition)

Each entry point YAML declares a `defaults:` list that pulls in sub-configs by directory name. The available hooks are:

| Hook (defaults key) | Config Directory | Purpose |
|---|---|---|
| `ner_model:` | `config/ner_model/` | Named-entity recognition model spec |
| `el_model:` | `config/el_model/` | Entity linking model spec |
| `openie_model:` | `config/openie_model/` | Open Information Extraction model spec |
| `graph_constructor:` | `config/graph_constructor/` | Knowledge graph construction pipeline spec |
| `sft_constructor:` | `config/sft_constructor/` | SFT training example constructor spec |
| `text_emb_model:` | `config/text_emb_model/` | Text embedding model spec |
| `qa_prompt:` | `config/qa_prompt/` | QA prompt template (per-dataset) |
| `qa_evaluator:` | `config/qa_evaluator/` | QA evaluation logic (per-dataset) |
| `agent_prompt:` | `config/agent_prompt/` | IRCoT agent prompt template (per-dataset) |
| `doc_ranker:` | `config/doc_ranker/` | Document ranker strategy for SFT model |
| `wandb:` | `config/wandb/` | Weights & Biases configuration |

### Filesystem I/O Contracts

| Path Pattern | Producer | Consumer |
|---|---|---|
| `data/*/raw/documents.json` | External dataset download | `index_dataset.py` / `GraphIndexer.index_data()` |
| `data/*/processed/stage1/{nodes,relations,edges}.csv` | `GraphIndexer` stage1 | `GraphIndexDataset.process_graph()` |
| `data/*/processed/stage2/{fingerprint}/{graph.pt,node2id.json,rel2id.json,train.pt,test.pt}` | `GraphIndexDataset.process_graph()` | Training scripts, `GFMRetriever` |
| `outputs/{kg_construction,kg_pretrain,qa_finetune,qa_inference,qa_agent_inference}/**/` | Hydra `hydra.run.dir` | Artifact storage (config snapshots, logs, checkpoints, predictions) |
| `outputs/*/pretrained/` | `utils.save_model_to_pretrained()` | `utils.load_model_from_pretrained()`, HuggingFace Hub |
| `outputs/*/prediction.jsonl` | QA inference scripts | Evaluator classes |

---

## Entry Point Reference

| File | Config Family | Config Name | Launch Command |
|---|---|---|---|
| `index_dataset.py` | `gfm_rag` | `index_dataset` | `python -m gfmrag.workflow.index_dataset` |
| `kgc_training.py` | `gfm_rag` | `kgc_training` | `python -m gfmrag.workflow.kgc_training` (or `torchrun`) |
| `sft_training.py` | `gfm_rag` | `sft_training` | `python -m gfmrag.workflow.sft_training` (or `torchrun`) |
| `qa.py` | `gfm_rag` | `qa_inference` | `python -m gfmrag.workflow.qa` |
| `qa_ircot_inference.py` | `gfm_rag` | `qa_ircot_inference` | `python -m gfmrag.workflow.qa_ircot_inference` |
| `experiments/visualize_path.py` | `gfm_reasoner` | `visualize_path` | `python -m gfmrag.workflow.experiments.visualize_path` |

The `gfm_reasoner/` config family provides equivalent YAML files for the second model family (G-reasoner-34M), reused by the same entry point scripts (only `config_path` differs — not shown in the decorator source but selectable via Hydra CLI override `--config-path=config/gfm_reasoner` or by editing the decorator).

---

## Key Files Summary

| File | Lines | Role |
|---|---|---|
| `__init__.py` | 0 | Empty package marker |
| `index_dataset.py` | 61 | Stage 1: raw document → knowledge graph indexing |
| `kgc_training.py` | 113 | Stage 2a: KGC contrastive pre-training |
| `sft_training.py` | 151 | Stage 2b: supervised fine-tuning with composite losses |
| `qa.py` | 148 | Stage 3a: single-step QA answer generation from pre-retrieved docs |
| `qa_ircot_inference.py` | 167 | Stage 3b: interleaved retrieval + LLM reasoning (IRCoT agent) |
| `experiments/visualize_path.py` | 130 | Path visualization for trained G-reasoner models |
| `config/` | ~15 directories | Hydra YAML configuration tree |
