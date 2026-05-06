# gfmrag/graph_index_datasets/

## Responsibility

**Data access and preprocessing layer** for graph-indexed datasets. This package owns the full lifecycle of graph-structured data within the GFM-RAG pipeline: raw CSV ingestion, entity/relation embedding generation (via pluggable text embedding models), fingerprint-based cache invalidation, serialization to PyTorch tensors, and on-demand multi-dataset loading with bounded memory. It serves as the single point of contact between persistent storage (stage1 raw files, stage2 processed artifacts) and all downstream consumers (retrieval, training, evaluation).

Two distinct concerns are separated into different classes:

- **`GraphIndexDataset` / `GraphIndexDatasetV1`** — single-dataset graph processing, serialization, and QA data preparation.
- **`GraphDatasetLoader`** — multi-dataset orchestration: LRU eviction, async prefetching, epoch-aware shuffling.

## Design Patterns

| Pattern | Where | How |
|---|---|---|
| **Template Method** | `GraphIndexDataset` → `GraphIndexDatasetV1` | `process_graph()` is overridden in V1 to filter edges by `target_type` and produce a sparse `target_to_other_types` mapping tensor. `attributes_to_text()` is also overridden (V1 returns only the node name; base class formats key:value pairs). Both call into shared helpers (`_read_csv_file`). |
| **Strategy** | `inverse_relation_feat` parameter | Two strategies for inverse relation embeddings selected at runtime: `"inverse"` uses the negative of the forward embedding (per [arXiv:2505.20422](http://arxiv.org/abs/2505.20422)); `"text"` encodes `"inverse_<name>"` through the text embedding model. Both produce `rel_emb` of shape `(2*num_relations, dim)`. |
| **Factory Method** | `hydra.utils.instantiate()` | `BaseTextEmbModel` instances are created on-demand from `DictConfig` inside `process_graph()` and `process_qa_data()`. The `text_emb_model_cfgs` subtree is passed through unmodified — the concrete model class is resolved via Hydra's `_target_` directive at the point of instantiation. |
| **LRU Cache** | `GraphDatasetLoader` | `loaded_datasets: OrderedDict[str, Any]` used as an LRU eviction map. `_get_dataset()` pops and re-inserts on access; `_manage_memory()` pops the oldest item (`popitem(last=False)`) when capacity is exceeded, followed by `gc.collect()`. |
| **Async Prefetching** | `GraphDatasetLoader` | `ProcessPoolExecutor` with `mp_context="spawn"` submits `_load_dataset_worker()` calls. A `threading.Lock` guards `loading_futures: dict[str, Future]`. Completed futures are drained in `_cleanup_completed_futures()`, called opportunistically before each synchronous `_get_dataset()` access. |
| **Content-Addressable Storage** | `GraphIndexDataset.fingerprint` | `self.fingerprint` is an MD5 hex digest of `(class_name + JSON(config))` where config includes all `FINGER_PRINT_ATTRS` plus the text-embedding model config (minus `batch_size`). The fingerprint determines the `stage2/{fingerprint}/` output directory, providing automatic cache invalidation when any relevant configuration changes. |
| **Data Object** | `GraphDataset` | A simple `@dataclass` with `name: str` and `data: Any`, yielded by `GraphDatasetLoader.__iter__()` to decouple iteration logic from the consumer. |

### Abstractions

- **`GraphIndexDataset`** is the base class; it defines the public interface (`process_graph()`, `load_graph()`, `load_qa_data()`, `process_qa_data()`, `attributes_to_text()`, `save_config()`) and the class-level constants (`FINGER_PRINT_ATTRS`, `RAW_GRAPH_NAMES`, `RAW_QA_DATA_NAMES`, `RAW_DOCUMENT_NAME`, `PROCESSED_GRAPH_NAMES`, `PROCESSED_QA_DATA_NAMES`).
- **`GraphIndexDatasetV1`** extends the base by adding `target_type` to the fingerprint and narrowing graph construction to edges where both source and target match that type. It also adds a sparse `target_to_other_types` tensor mapping target-type nodes to other-type nodes via existing edges.
- **`GraphDatasetLoader`** is not part of the class hierarchy — it is a standalone iterator that wraps `GraphIndexDataset` construction. It depends on `datasets_cfg._target_` (resolved via `hydra.utils.get_class()`) to determine which `GraphIndexDataset` subclass to instantiate per dataset name.

## Data & Control Flow

### Single-dataset flow (GraphIndexDataset)

```
Constructor __init__(root, data_name, text_emb_model_cfgs, ...)
  │
  ├─ Compute fingerprint (MD5 of class_name + config)
  │
  ├─ load_graph(force_reload)
  │   ├─ Check files_exist(processed_graph) ← [graph.pt, node2id.json, rel2id.json]
  │   ├─ If missing: os.makedirs(processed_dir) → process_graph()
  │   ├─ Load: torch.load(graph.pt), json.load(node2id.json), json.load(rel2id.json)
  │   └─ Set: self.graph, self.node2id, self.rel2id, self.id2node, self.feat_dim
  │
  ├─ load_qa_data(graph_rebuild, force_reload)
  │   ├─ For each raw QA file (train.json / test.json) check if processed .pt exists
  │   ├─ If missing or forced: process_qa_data(need_to_process_qa_data)
  │   ├─ Load: torch.load(train.pt), torch.load(test.pt)
  │   └─ Set: self.train_data, self.test_data, self.raw_train_data, self.raw_test_data, self.doc
  │
  └─ If any rebuild: save_config() → config.json in processed_dir


process_graph()  [skeleton — overridden in V1]
  │
  ├─ Read nodes.csv  → _read_csv_file() → nodes_df (indexed by uid or name, with id column)
  │   ├─ Factorize node types → node_types tensor, node_type_names
  │   ├─ Build nodes_by_type dict {type_name: LongTensor of ids}
  │
  ├─ Read relations.csv → _read_csv_file() → relations_df
  │   └─ rel2id = relations_df["id"].to_dict()
  │
  ├─ Read edges.csv → edges_df
  │   ├─ Map source/target/relation columns → u/v/r IDs
  │   ├─ Drop rows with missing mappings (log warning)
  │   ├─ Build (u, v, r) edge tuples
  │   └─ Create train_target_edges (2 x E tensor) and train_target_etypes (E tensor)
  │
  ├─ Concatenate inverse edges:
  │   train_edges = [target_edges, target_edges.flip(0)]  (2 x 2E)
  │   train_etypes = [target_etypes, target_etypes + num_relations]  (2E)
  │
  ├─ Save node2id.json, rel2id.json (with inverse_* entries added)
  │
  ├─ Instantiate text_emb_model via hydra.utils.instantiate(text_emb_model_cfgs)
  │
  ├─ Generate relation embeddings (if use_relation_feat):
  │   ├─ Encode relation names → rel_emb (num_relations x dim)
  │   ├─ Apply inverse strategy: "inverse" → cat(rel_emb, -rel_emb), "text" → cat(rel_emb, inverse_rel_emb)
  │
  ├─ Generate node embeddings (if use_node_feat):
  │   └─ Encode "name: X\ntype: Y\nattr: Z" → node_emb (num_nodes x dim)
  │
  ├─ Generate edge embeddings (if use_edge_feat):
  │   └─ Encode edge attributes → edge_emb (num_edges x dim)
  │
  ├─ Determine feat_dim from first non-None embedding; 0 if all None
  │
  └─ Build torch_geometric.data.Data(edge_index, edge_type, x, rel_attr, ...) → save graph.pt


process_qa_data(qa_data_names)
  │
  ├─ For each sample in each raw JSON:
  │   ├─ Map start_node/target_node strings to IDs via self.node2id
  │   ├─ entities_to_mask(start_nodes, num_nodes) → multi-hot mask tensor
  │   ├─ entities_to_mask(target_nodes, num_nodes) → multi-hot mask tensor
  │   ├─ Collect question string
  │   └─ Skip if start_nodes empty or (skip_empty_target and target_nodes empty)
  │
  ├─ Encode all questions via text_emb_model.encode(is_query=True) → question_embeddings
  ├─ Stack masks → (num_samples, num_nodes) tensors
  ├─ Build HuggingFace datasets.Dataset → convert to torch format
  └─ Split by data_name, save each split as torch.utils.data.Subset → train.pt / test.pt
```

### V1-specific additions to process_graph()

```
  V1 adds the following between edge loading and inverse-edge construction:
  │
  ├─ Annotate each edge with source_type, target_type from nodes_df
  ├─ Filter to edges where source_type == target_type == self.target_type
  │   (edges within the target type only)
  │
  └─ Build target_to_other_types:
      For each other_type reachable from target_type via valid_edges_df:
        Create sparse_coo_tensor(indices=[u, v], values=1, size=(num_nodes, num_nodes))
```

### Multi-dataset flow (GraphDatasetLoader)

```
GraphDatasetLoader.__init__(datasets_cfg, data_names, shuffle, max_datasets_in_memory, data_loading_workers)
  │
  ├─ Initialize: OrderedDict for LRU cache, ProcessPoolExecutor (spawn), loading_futures dict
  │
  └─ Note: No datasets are loaded at construction time. Loading is lazy.


GraphDatasetLoader.__iter__()  →  Generator[GraphDataset]
  │
  ├─ Copy + optionally shuffle data_names with np.random.seed(epoch)
  │
  ├─ _preload_datasets():
  │   ├─ Synchronously load first max_datasets_in_memory items (blocking)
  │   └─ _start_async_loading() for the next batch (up to data_loading_workers)
  │
  ├─ For each data_name in order:
  │   ├─ Determine next indices → _get_next_datasets_to_prefetch()
  │   ├─ _start_async_loading(next_batch)
  │   │   ├─ Each future calls _load_dataset_worker() in a child process:
  │   │   │   ├─ OmegaConf.create(datasets_cfg) from serialized dict
  │   │   │   ├─ get_class(datasets_cfg._target_)(cfgs, data_name=data_name)
  │   │   │   └─ Returns constructed GraphIndexDataset (or None on error)
  │   │   └─ Future stored in self.loading_futures[data_name]
  │   │
  │   └─ _get_dataset(data_name):
  │       ├─ _cleanup_completed_futures(): drain done futures into loaded_datasets
  │       ├─ If in LRU cache: pop + re-insert (update LRU order), return
  │       ├─ If in loading_futures: _wait_for_dataset() (blocking, 30s timeout), cache, return
  │       └─ Otherwise: load synchronously, evict if needed, cache, return
  │
  └─ Yields GraphDataset(name=data_name, data=dataset)


Key supporting methods:
  ├─ _manage_memory(): While len(loaded) >= max_datasets_in_memory, popitem(last=False) + del + gc.collect()
  ├─ wait_for_all_loading(timeout): Busy-poll loading_futures until empty or timeout (0.1s interval)
  ├─ clear_cache(): Cancel all futures, clear dicts, gc.collect()
  ├─ get_memory_info(): Returns dict of counts and names for monitoring
  ├─ set_epoch(epoch): Seeds numpy for reproducible shuffling
  └─ shutdown(): Cancel pending futures, shutdown executor
```

### State transitions (loader lifecycle)

```
INIT ──→ PRELOADING ──→ ITERATING ──→ (repeat for each data_name) ──→ EXHAUSTED
              │                            │
              └── sync load first N ────────┘
                + async prefetch next batch     └── LRU eviction as needed
```

## Integration Points

### Consumer modules (import from this package)

| Module | Import | Usage |
|---|---|---|
| `gfmrag.gfmrag_retriever` | `GraphIndexDataset` | `GFMRetriever.from_index()` constructs a dataset instance via `hydra.utils.instantiate()`, then accesses `.graph`, `.node2id`, `.rel2id`, `.train_data`, `.test_data`. Captures `_STAGE1_GRAPH_NAMES = GraphIndexDataset.RAW_GRAPH_NAMES` at import time for test patching. |
| `gfmrag.graph_indexer` | `GraphIndexDataset` | `GraphIndexer.index_data()` accesses `GraphIndexDataset.RAW_GRAPH_NAMES` to validate stage1 file existence. |
| `gfmrag.graph_index_construction.graph_constructors.kg_constructor` | `GraphIndexDataset` | Uses `GraphIndexDataset.RAW_GRAPH_NAMES` and `GraphIndexDataset.RAW_DOCUMENT_NAME` to verify construction output paths. |
| `gfmrag.graph_index_construction.sft_constructors.gfm_rag_constructor` | `GraphIndexDataset` | Same as above — references `RAW_GRAPH_NAMES` constant. |
| `gfmrag.graph_index_construction.sft_constructors.gfm_reasoner_constructor` | `GraphIndexDataset` | Same as above. |
| `gfmrag.graph_index_construction.sft_constructors.hipporag2_constructor` | `GraphIndexDataset` | Same as above. |
| `gfmrag.utils.util` | `GraphIndexDataset` | Type-checks that a class is a subclass of `GraphIndexDataset` before using it. |
| `gfmrag.workflow.kgc_training` | `GraphDatasetLoader` | Creates `train_dataset_loader = GraphDatasetLoader(...)` for KGC training loop. |
| `gfmrag.workflow.sft_training` | `GraphDatasetLoader` | Creates `train_graph_dataset_loader` and `valid_graph_dataset_loader` for SFT training loop. |
| `gfmrag.trainers.base_trainer` | `GraphDatasetLoader` | Accepts `train_graph_dataset_loader` / `eval_graph_dataset_loader` parameters; calls `set_epoch()` before each epoch and iterates during training. |
| `gfmrag.trainers.kgc_trainer` | `GraphDatasetLoader` | Same pattern — passes loader to base trainer, iterates over `GraphDataset` objects. |
| `gfmrag.trainers.sft_trainer` | `GraphDatasetLoader` | Same pattern. |
| `tests.test_gfmrag_retriever` | `GraphIndexDataset` | Patches `GraphIndexDataset` to bypass full file I/O during retriever unit tests. |
| `tests.test_stage2_dataset` | `GraphIndexDataset`, `GraphIndexDatasetV1` | End-to-end integration tests for stage2 processing. |
| `tests.test_pretrained_dataset_config` | `GraphIndexDataset`, `GraphIndexDatasetV1` | Serialization round-trip tests for `export_config_dict()`. |

### Dependencies (imported by this package)

| Dependency | Used by | Purpose |
|---|---|---|
| `torch_geometric.data.Data` | `GraphIndexDataset`, `GraphIndexDatasetV1` | Graph data container (edge_index, edge_type, x, rel_attr, etc.) |
| `torch.utils.data.Dataset` / `Subset` | `GraphIndexDataset` | QA data split serialization |
| `hydra.utils.instantiate` / `get_class` | Both | Dynamic instantiation of text embedding models and dataset classes from OmegaConf config |
| `omegaconf.DictConfig` / `OmegaConf` | Both | Configuration tree traversal, serialization, and deserialization |
| `datasets.Dataset` (HuggingFace) | `GraphIndexDataset` | Temporary in-memory structured dataset before splitting to `torch.Subset` |
| `pandas` | Both | CSV I/O and groupby operations |
| `BaseTextEmbModel` (`gfmrag.text_emb_models`) | Both | Text-to-embedding encoding interface (`.encode(texts, is_query)`) |
| `entities_to_mask` (`gfmrag.utils.qa_utils`) | `GraphIndexDataset` | Converts entity ID lists to multi-hot binary masks |
| `get_rank` (`gfmrag.utils`) | Both | Distributed rank logging |
| `concurrent.futures.ProcessPoolExecutor` | `GraphDatasetLoader` | Async dataset loading in child processes |
| `threading.Lock` | `GraphDatasetLoader` | Thread-safe access to `loading_futures` |

### API surface (public exports via `__init__.py`)

```python
__all__ = ["GraphDatasetLoader", "GraphIndexDataset", "GraphIndexDatasetV1"]
```

### Configuration contract

The `datasets_cfg` `DictConfig` consumed by `GraphDatasetLoader` and passed via `_load_dataset_worker()` is expected to have the structure:

```yaml
_target_: "gfmrag.graph_index_datasets.GraphIndexDataset"  # or subclass FQN
cfgs:
  root: str
  text_emb_model_cfgs:
    _target_: "gfmrag.text_emb_models.SomeModel"
    # ... model-specific params (batch_size is stripped for fingerprinting)
  # Optional overrides:
  use_node_feat: bool       # default True
  use_relation_feat: bool   # default True
  use_edge_feat: bool       # default False
  inverse_relation_feat: str  # "text" | "inverse"
  skip_empty_target: bool   # default True
  # GraphIndexDatasetV1 only:
  target_type: str
```

### File format contracts

**Input (stage1 / raw):**
- `nodes.csv` — Must have `uid` or `name` column (unique), `type` column, optional `attributes` (JSON dict in string form).
- `relations.csv` — Same structure; `name` column for the relation label.
- `edges.csv` — Columns: `source`, `target`, `relation` (string references to node/relation identifiers), optional `attributes`.
- `train.json` / `test.json` — Array of `{id, question, start_nodes: {type: [id, ...]}, target_nodes: {type: [id, ...]}}`.
- `documents.json` — Generic JSON blob (opaque to this module, loaded and stored as `self.doc`).

**Output (stage2 / processed):**
- `graph.pt` — `torch_geometric.data.Data` with attributes: `node_type`, `node_type_names`, `nodes_by_type`, `edge_index`, `edge_type`, `num_nodes`, `target_edge_index`, `target_edge_type`, `num_relations`, `x`, `rel_attr`, `edge_attr`, `feat_dim`. V1 additionally includes `target_to_other_types` (dict of sparse tensors).
- `node2id.json` — `{node_uid: int_id}`.
- `rel2id.json` — `{rel_name: int_id}` including `inverse_*` entries.
- `train.pt` / `test.pt` — `torch.utils.data.Subset` over a HuggingFace `Dataset` with fields: `question_embeddings`, `start_nodes_mask`, `target_nodes_mask`, `id`.
- `config.json` — Persisted dataset configuration (model class, text_emb_model_cfgs, fingerprint attrs).
