# gfmrag/utils/

## Responsibility

The `gfmrag/utils/` package provides **cross-cutting infrastructure utilities** for the GFM-RAG framework. It is a **support library** that delivers five orthogonal concerns consumed by every major subsystem (trainers, workflows, datasets, constructors, and the retriever entrypoint):

| Concern | Module | Role |
|---|---|---|
| Distributed orchestration | `setup_training.py` | Rank discovery, process group init, device resolution, log suppression, barrier sync |
| Model serialization | `util.py` | Hydra-instantiated model save/load with bundled dataset config |
| Metric computation | `qa_utils.py` | Ranking metrics (MRR, Recall@k, Hits@k, MAPE), distributed result gathering |
| Graph partitioning | `dist_graph_utils.py` | Split a `torch_geometric.data.Data` graph across GPUs for SPMD message passing |
| Experiment tracking | `wandb_utils.py` | Weights & Biases init, metric logging, checkpoint artifact upload |

The package explicitly re-exports selected symbols through `__init__.py`; most consumers import via `from gfmrag import utils` then use `utils.<function>()`.

---

## Design Patterns

### 1. Facade (through `__init__.py`)
The `__init__.py` re-exports `partition_graph_edges` directly and uses wildcard imports (`*`) from the four other modules. This gives callers a single `gfmrag.utils` namespace, though in practice most internal consumers bypass it and import submodules directly (e.g. `from gfmrag.utils.dist_graph_utils import partition_graph_edges`).

### 2. Template Method (implicit in model serialization — `util.py`)
`save_model_to_pretrained()` follows a fixed four-step sequence:
1. Create output directory
2. Serialise model config via `OmegaConf.to_container`
3. Fetch dataset class via `hydra.utils.get_class` and call its `export_config_dict`
4. Write `config.json` + `model.pth`

`load_model_from_pretrained()` inverts the same sequence:
1. Resolve `config.json` via HuggingFace `cached_file`
2. Deserialise config
3. Call `hydra.utils.instantiate(config)` to reconstruct the model
4. Load state dict from `model.pth`

Both functions rely on `HydraUtils.get_class` / `HydraUtils.instantiate` — a **Factory** supplied by the Hydra framework.

### 3. Strategy (precision configuration — `setup_training.py`)
`configure_model_precision(model, device, precision)` implements a **Strategy** pattern where the `precision` argument selects one of four policies:
- `"auto"` → probe `torch.cuda.is_bf16_supported()`, fall back FP16
- `"bfloat16"` → try BF16, fall back FP16
- `"float16"` → unconditional FP16
- `"float32"` → no AMP, retain FP32

Each path returns `(model, dtype)` — a uniform interface for downstream AMP usage.

### 4. Envoy / Proxy (distributed gather — `qa_utils.py`)
`gather_results()` implements a **reduction-collective** pattern: each rank scatters its local ranking tensors into pre-allocated buffers sized by `all_reduce(SUM)` on element counts, then calls `dist.all_reduce(SUM)` to reconstruct the global result. This is an **AllGather emulation** via `all_reduce(SUM)` where each rank writes to disjoint buffer regions.

### 5. Adapter (distributed graph partitioning — `dist_graph_utils.py`)
`partition_graph_edges()` and `partition_graph_metis()` both adapt a single-device `torch_geometric.data.Data` graph into a partitioned `Data` object annotated with a `dist_context` tuple and a `boundary_mode` flag. The output contract is consumed by `QueryNBFNet.bellmanford` in `gfmrag/models/ultra/models.py`, which checks for the `dist_context` attribute to decide whether to run SPMD message passing.

### 6. Singleton (wandb run — `wandb_utils.py`)
The `wandb` module-level singleton is guarded at every call site: every function (`log_metrics`, `log_model_checkpoint`, `finish_wandb`, `watch_model`) checks `if not wandb.run` before executing, making it safe to call from distributed workers where only rank 0 initialises W&B.

---

## Data & Control Flow

### A. Model Persistence Flow (`util.py`)

```
save_model_to_pretrained(model, cfg, path)
  │
  ├─ OmegaConf.to_container(cfg.model)         # flatten model config
  ├─ get_class(cfg.datasets._target_)           # resolve dataset class
  ├─ dataset_cls.export_config_dict(cfg.datasets.cfgs)  # export dataset config
  ├─ json.dump(..., "config.json")
  └─ torch.save({"model": model.state_dict()}, "model.pth")

load_model_from_pretrained(path)
  │
  ├─ cached_file(path, "config.json")           # HuggingFace file resolution
  ├─ json.load(config.json)
  ├─ instantiate(config["model_config"])         # Hydra instantiation
  ├─ cached_file(path, "model.pth")
  ├─ torch.load("model.pth", weights_only=True)
  └─ model.load_state_dict(state["model"])
```

State transitions: **Hydra DictConfig → serialised JSON → flattened dict → PyTorch state dict**.

### B. Training Initialisation Flow (`setup_training.py`)

```
init_distributed_mode(timeout)
  │
  ├─ get_world_size()          # env RANK / dist.get_rank()
  ├─ torch.cuda.set_device(     # assign GPU by LOCAL_RANK
  │      get_local_rank())
  ├─ dist.init_process_group(   # NCCL backend, env://
  │      "nccl",
  │      init_method="env://",
  │      device_id=get_device())
  ├─ synchronize()              # dist.barrier()
  └─ setup_for_distributed()    # monkey-patch builtins.print
```

Supporting queries:
- `get_rank()` → fallback chain: `dist.get_rank()` → `$RANK` → `0`
- `get_world_size()` → fallback chain: `dist.get_world_size()` → `$WORLD_SIZE` → `1`
- `get_device()` → `torch.device(LOCAL_RANK)` if CUDA, else `"cpu"`

### C. Metric Computation Flow (`qa_utils.py`)

```
batch_evaluate(pred_logits, target_mask)
  │
  ├─ Sort pred_logits descending → get global ranking per entity
  ├─ variadic.native_scatter → target_ranking for every positive
  ├─ variadic.variadic_sort → order among answers per query
  ├─ Filter to "hard" answers via multi_slice_mask
  └─ Return (filtered_ranking, global_target_ranking)

evaluate(pred, target, metrics=["mrr", "recall@k", "hits@k", "mape"])
  │
  └─ For each metric name → compute per-query score → variadic_mean/sum

gather_results(pred, target, rank, world_size, device)
  │
  ├─ all_reduce(SUM) on element counts per rank
  ├─ Scatter local tensors into disjoint global buffer regions
  ├─ all_reduce(SUM) on full buffers
  └─ Return (global_ranking, global_target)
```

### D. Graph Partitioning Flow (`dist_graph_utils.py`)

```
partition_graph_edges(graph, rank, world_size)
  │
  ├─ Ceiling-divide nodes: each rank gets ceil(N/world_size)
  ├─ Filter edge_index[:, mask] where target (edge_index[0]) ∈ local slice
  ├─ Clone graph, attach dist_context=(rank, world_size)
  └─ Return filtered Data

partition_graph_metis(graph, rank, world_size)
  │
  ├─ (Rank 0) Build adjacency list, call pymetis.part_graph
  ├─ dist.broadcast_object_list → all ranks get node2part
  ├─ Select local nodes = (node2part == rank)
  ├─ Filter edges: keep those whose target node is owned
  ├─ Identify boundary nodes (remote sources needed by local edges)
  ├─ Build compact index space: [0, local_N) local + [local_N, compact) boundary
  └─ Return annotated Data with boundary_mode=True
```

Edge convention note: `edge_index[0]` = **destination** (output), `edge_index[1]` = **source** (input) — reversed from standard PyG, matching the `rspmm` CUDA kernel.

### E. W&B Logging Flow (`wandb_utils.py`)

```
init_wandb(cfg)       → wandb.init()
watch_model(model)    → wandb.watch()
log_metrics(...)      → wandb.log()
log_model_checkpoint(...) → wandb.Artifact → wandb.log_artifact()
finish_wandb()        → wandb.finish()
```

All functions are **idempotent to missing wandb.run** — they silently return if W&B was never initialised (safe for non-master ranks).

---

## Integration Points

### Consumer Modules

| Consumer | Imported Symbols | File |
|---|---|---|
| `gfmrag.gfmrag_retriever` | `utils.load_model_from_pretrained`, `entities_to_mask` | `gfmrag_retriever.py` |
| `gfmrag.graph_indexer` | `check_all_files_exist` | `graph_indexer.py` |
| `gfmrag.graph_index_datasets.graph_index_dataset` | `get_rank`, `entities_to_mask` | `graph_index_dataset.py` |
| `gfmrag.graph_index_construction.sft_constructors.*` | `check_all_files_exist` | 3 constructor files |
| `gfmrag.trainers.base_trainer` | `wandb_utils.*`, `setup_training.*` | `base_trainer.py` |
| `gfmrag.trainers.sft_trainer` | `partition_graph_edges`, `partition_graph_metis`, `qa_utils.*` | `sft_trainer.py` |
| `gfmrag.trainers.kgc_trainer` | `qa_utils.evaluate` (via `utils`) | `kgc_trainer.py` |
| `gfmrag.workflow.sft_training` | `util.*`, `setup_training.*`, `wandb_utils.*` | `sft_training.py` |
| `gfmrag.workflow.kgc_training` | `util.*`, `setup_training.*`, `wandb_utils.*` | `kgc_training.py` |
| `gfmrag.workflow.qa` | `setup_training.*` | `qa.py` |
| `gfmrag.workflow.experiments.visualize_path` | `load_model_from_pretrained`, `setup_training.*` | `visualize_path.py` |
| `tests.test_pretrained_dataset_config` | `save_model_to_pretrained` | `test_pretrained_dataset_config.py` |

### Dependencies (3rd-party)

| Dependency | Used By | Integration Mechanism |
|---|---|---|
| `pymetis` | `dist_graph_utils.py` | `pymetis.part_graph(world_size, adjacency)` |
| `torch.distributed` | `dist_graph_utils.py`, `qa_utils.py`, `setup_training.py` | `dist.init_process_group`, `all_reduce`, `broadcast_object_list`, `barrier` |
| `torch_geometric.data.Data` | `dist_graph_utils.py` | Graphs with `.edge_index`, `.edge_type`, `.num_nodes` |
| `hydra.utils` (`get_class`, `instantiate`) | `util.py` | Dynamic class resolution and config-driven instantiation |
| `omegaconf.DictConfig` / `OmegaConf` | `util.py`, `wandb_utils.py` | Config serialisation / deserialisation |
| `transformers.utils.cached_file` | `util.py` | HuggingFace Hub file resolution for pretrained paths |
| `wandb` | `wandb_utils.py` | `wandb.init`, `wandb.log`, `wandb.Artifact`, `wandb.watch` |
| `gfmrag.models.ultra.variadic` | `qa_utils.py` | `variadic_mean`, `variadic_sum`, `variadic_sort`, `variadic_arange`, `native_scatter`, `multi_slice_mask` |

### Internal API Contract Surface

| Symbol | Signature | Contract |
|---|---|---|
| `partition_graph_edges` | `(graph: Data, rank: int, world_size: int) -> Data` | Attaches `.dist_context` |
| `partition_graph_metis` | `(graph: Data, rank: int, world_size: int) -> Data` | Attaches `.dist_context`, `.local_nodes`, `.boundary_nodes`, `.compact_size`, `.node2part`, `.boundary_mode=True` |
| `evaluate` | `(pred, target, metrics: list[str]) -> dict[str, float]` | Returns `{mrr: ..., recall@k: ..., hits@k: ..., mape: ...}` |
| `gather_results` | `(pred, target, rank, world_size, device) -> tuple` | Returns `(all_ranking, all_num_pred), (all_answer_ranking, all_num_target)` |
| `batch_evaluate` | `(pred, target, limit_nodes=None) -> tuple` | Returns `(ranking, target_ranking)` tensors |
| `load_model_from_pretrained` | `(path: str) -> tuple[nn.Module, dict]` | Model loaded via `hydra.utils.instantiate` |
| `save_model_to_pretrained` | `(model, cfg: DictConfig, path: str) -> None` | Writes `config.json` + `model.pth` |
| `init_multi_dataset` | `(cfg: DictConfig, world_size: int, rank: int) -> list[int]` | Returns list of `feat_dim` per dataset |
| `init_distributed_mode` | `(timeout=None) -> None` | Initialises NCCL world, suppresses non-master prints |
| `configure_model_precision` | `(model, device, precision: str) -> tuple[nn.Module, torch.dtype]` | Returns model configured for AMP |
| `init_wandb` | `(cfg, project_name, run_name, tags) -> None` | Safe no-op if `cfg.wandb.enabled=False` |
| `log_metrics` | `(metrics, step, prefix) -> None` | No-op if `wandb.run` is `None` |
