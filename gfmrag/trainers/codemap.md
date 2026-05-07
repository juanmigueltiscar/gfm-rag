# `gfmrag/trainers/` — Training Loop Orchestration

## Responsibility

Provides the **training loop abstraction** for the GFM-RAG model family. This directory implements the **Template Method** pattern for supervised training workflows: a `BaseTrainer` defines the skeleton of epoch-based training (data iteration, gradient scaling, checkpointing, metric logging, distributed synchronization), while concrete subclasses override the task-specific steps (`train_step`, `_create_task_dataset`, `evaluate`). The module acts as an **orchestrator** between graph datasets, model forward passes, loss computation, and metric evaluation — it does **not** define model architectures or data pipelines but wires them together.

Two concrete training regimes are supported:

- **KGC (Knowledge Graph Completion) pretraining** — `KGCTrainer`
- **SFT (Supervised Fine-Tuning) for QA/retrieval** — `SFTTrainer`

Auxiliary types: `TrainingArguments` (hyperparameter dataclass), `TaskDataset` (task-scoped data container), `SFTLoss` (composite loss descriptor).

---

## Design Patterns

| Pattern | Where | How |
|---------|-------|-----|
| **Template Method** | `BaseTrainer.train()` / `_train_epoch()` | Defines the invariant training skeleton; subclasses supply `train_step()`, `_create_task_dataset()`, `evaluate()`. |
| **Strategy** | `SFTTrainer.loss_functions: list[SFTLoss]` | Plug-composable loss functions selected at construction time. Each `SFTLoss` wraps a `BaseLoss` instance with a node-type target, weight, and distillation flag. The trainer iterates the list and sums weighted losses. |
| **Abstract Factory** | `_create_task_dataset()` | Each subclass manufactures its own `TaskDataset` (or `PretrainTaskDataset`) populating the dataloader, graph reference, and any subclass-specific fields (e.g., `val_filtered_graph` in KGC). |
| **Data Mapper** | `GraphDatasetLoader` iteration | The trainer consumes `GraphDataset` namedtuples from the loader, mapping each to a `TaskDataset` via `_create_task_dataset()` before iterating its inner `DataLoader`. |
| **Singleton-like (per-process)** | `utils.get_device()`, `utils.get_rank()`, `utils.get_world_size()` | Rank-aware helpers used throughout to gate logging, checkpointing, and distributed collectives. |
| **Command pattern** | `TrainingArguments` | Encapsulates all hyperparameters and flags (`do_train`, `do_eval`, `eval_strategy`, `split_graph_training`, etc.) as a single immutable dataclass consumed by the trainer. |

### Abstractions & Interfaces

```
BaseTrainer (ABC)                        ← gfmrag/utils, gfmrag/utils/wandb_utils
├── _create_task_dataset(GraphDataset, is_train) → TaskDataset    [abstract]
├── train_step(batch, TaskDataset)       → dict[str, float|Tensor] [abstract]
├── evaluate()                           → dict[str, float]       [abstract]
├── train()                              → None                   [concrete, template]
├── _train_epoch()                       → None                   [concrete]
├── _save_checkpoint() / _load_checkpoint()                        [concrete]
├── _maybe_save_best_model()                                      [concrete]
└── _log_metrics()                                                 [concrete]

KGCTrainer(BaseTrainer)
├── _create_task_dataset() → PretrainTaskDataset (adds val_filtered_graph)
├── train_step()            → negative-sampling + BCE + adversarial weighting
└── evaluate()              → filtered ranking (MR, MRR, Hits@K) with all-reduce

SFTTrainer(BaseTrainer)
├── _create_task_dataset()  → TaskDataset (split-graph-aware sampler selection)
├── train_step()            → per-node-type loss accumulation + optional distillation
├── evaluate()              → per-type ranking metrics (MRR, Hits@K) + gather_results
└── predict()               → top-k prediction with id→node mapping + all_gather_object
```

### Key Dataclass Types

```python
@dataclass
class TaskDataset:
    name: str
    graph: Any  # PyG Data object
    data_loader: DataLoader


@dataclass(kw_only=True)
class PretrainTaskDataset(TaskDataset):
    val_filtered_graph: Data  # filtered graph for KGC ranking


@dataclass
class SFTLoss:
    name: str
    loss_fn: BaseLoss
    weight: float
    target_node_type: str
    is_distillation_loss: bool | None = False
```

---

## Data & Control Flow

### Training entry (`BaseTrainer.train()`)

```
1.  BaseTrainer.__init__()
      └─ _setup_model()
           ├─ _load_checkpoint()           ← if args.resume_from_checkpoint
           ├─ utils.configure_model_precision()  → AMP scaler / dtype
           └─ DDP wrap (if world_size > 1)
      state = {epoch, global_step, best_metric, best_epoch}

2.  BaseTrainer.train()
      ├─ for epoch in [start_epoch .. num_epoch):
      │    ├─ self._train_epoch()
      │    │    ├─ for graph_dataset in train_graph_dataset_loader:
      │    │    │    ├─ task_dataset = self._create_task_dataset(graph_dataset, is_train=True)
      │    │    │    ├─ for batch in data_loader (optionally islice'd):
      │    │    │    │    ├─ self.train_step(batch, task_dataset)  → dict{"loss", ...}
      │    │    │    │    ├─ scaler.scale(loss).backward()
      │    │    │    │    ├─ [if split_graph_training: all_reduce(grad)]
      │    │    │    │    ├─ scaler.step(optimizer); scaler.update()
      │    │    │    │    ├─ optimizer.zero_grad()
      │    │    │    │    ├─ self._log_metrics()    ← every logging_steps
      │    │    │    │    └─ [if eval_strategy=step: evaluate() + _maybe_save_best_model()]
      │    │    │    └─ global_step++
      │    │    └─ log epoch averages to wandb
      │    ├─ [if eval_strategy=epoch: evaluate() + _maybe_save_best_model()]
      │    └─ self._save_checkpoint()
      ├─ [if load_best_model_at_end: _load_checkpoint(model_best.pth)]
      └─ [if do_eval: evaluate() → log as "final/"]
```

### KGC-specific data flow (`KGCTrainer`)

```
_create_task_dataset()
  Input:  GraphDataset → graph (Data with target_edge_index/type, num_nodes)
  └─ Constructs PretrainTaskDataset:
       ├─ val_filtered_graph = Data(edge_index, edge_type, num_nodes)
       ├─ triples = [head, tail, rel] matrix
       └─ DataLoader(triples, sampler=DistributedSampler)

train_step()
  Input: batch [B, 3], task_dataset.graph
  └─ tasks.negative_sampling(graph, batch, num_negative) → [B, 1+neg, 3]
  └─ self.parallel_model(graph, batch) → logits [B, 1+neg]
  └─ BCEWithLogitsLoss + adversarial softmax weighting → {"loss": scalar}

evaluate()
  └─ For each eval GraphDataset:
       ├─ _create_task_dataset(is_train=False)
       ├─ tasks.all_negative(graph, batch) → head-replacement & tail-replacement batches
       ├─ strict_negative_mask(filtered_data, batch)   ← filters known true triples
       ├─ tasks.compute_ranking() → ranks per triple
       ├─ all_reduce across ranks (cumulative ranking tensor)
       └─ MR / MRR / Hits@K (filtered, optionally tail-only, optionally unbiased)
```

### SFT-specific data flow (`SFTTrainer`)

```
_create_task_dataset()
  Input: GraphDataset → sft_dataset (train_data or test_data)
  └─ Sampler selection:
       ├─ split_graph_training → SequentialSampler (all ranks see same batch)
       └─ else → DistributedSampler (shuffle=is_train)
  └─ [if split_graph_training: partition_graph_metis/edges() → subgraph per rank]
  └─ returns TaskDataset(graph, data_loader)

train_step()
  Input: batch (dict with target_nodes_mask, question_embeddings), task_dataset.graph
  └─ self.parallel_model(graph, batch) → scores per node
  └─ For each SFTLoss in self.loss_functions:
       ├─ target_node_pred = pred[:, target_node_ids]
       ├─ [if distillation: target = question_emb @ target_node_emb.T]
       ├─ [else: target = target_nodes_mask[:, target_node_ids]]
       └─ loss_fn(pred, target) → weighted sum → total_loss
  └─ returns {"loss": total_loss, ...per-loss names}

evaluate()
  └─ For each eval GraphDataset (split-graph-aware):
       ├─ gather_results() across ranks (if not split) → full ranking tensors
       └─ utils.evaluate(rankings, targets, metrics) → node_type_metric dict
  └─ averages watched metric (default: document_mrr) across datasets
```

### State transitions

```
┌─────────────┐     train() called      ┌───────────┐
│  __init__   │ ──────────────────────> │  Training  │
│  (model,    │                         │  Loop      │
│   optim,    │     epoch loop          │  (epochs)  │
│   loaders)  │ ──────────────────────> │            │
└─────────────┘                         └─────┬─────┘
                                              │
                           ┌──────────────────┼──────────────────┐
                           ▼                  ▼                  ▼
                     eval_strategy       eval_strategy       checkpoint
                     = "epoch"           = "step"            (periodic)
                           │                  │
                           ▼                  ▼
                     _maybe_save_best_model() → model_best.pth
                           │
                     load_best_model_at_end → final evaluate
```

---

## Integration Points

### Dependencies (imported by this module)

| Dependency | Used By | Role |
|-----------|---------|------|
| `gfmrag.utils` | `BaseTrainer`, `KGCTrainer`, `SFTTrainer` | `get_device()`, `get_rank()`, `get_world_size()`, `is_main_process()`, `synchronize()`, `configure_model_precision()`, `batch_evaluate()`, `evaluate()`, `gather_results()` |
| `gfmrag.utils.wandb_utils` | `BaseTrainer` | `log_metrics()`, `log_model_checkpoint()` — W&B logging hooks |
| `gfmrag.utils.dist_graph_utils` | `SFTTrainer` | `partition_graph_edges()`, `partition_graph_metis()` — split-graph partitioning |
| `gfmrag.graph_index_datasets.graph_dataset_loader` | All | `GraphDataset`, `GraphDatasetLoader` — input data abstraction |
| `gfmrag.losses` | `SFTTrainer` | `BaseLoss` — pluggable loss function interface |
| `gfmrag.models.ultra.tasks` | `KGCTrainer` | `negative_sampling()`, `all_negative()`, `strict_negative_mask()`, `compute_ranking()` |
| `gfmrag.models.ultra.query_utils` | `SFTTrainer` | `cuda()`, `cat()` |
| `torch` + `torch.nn` | All | DDP, AMP, DataLoader, DistributedSampler, optimizers |
| `torch.distributed` | `BaseTrainer`, `KGCTrainer` | `all_reduce`, `all_gather_object` |
| `torch_geometric.data.Data` | `KGCTrainer` | Graph data container for filtered evaluation |

### Consumers (modules that import from trainers)

| Consumer | What it uses |
|----------|-------------|
| `gfmrag.workflow.*` (Hydra entrypoints) | `KGCTrainer`, `SFTTrainer`, `TrainingArguments` — instantiated via `instantiate()` from Hydra configs |
| `gfmrag.__init__` / external API | Likely via re-export from `gfmrag.workflow` |

### Hooks & extension points

| Hook | Location | Extensibility |
|------|----------|--------------|
| `train_step()` | Abstract on `BaseTrainer` | Required override — defines forward + loss per batch |
| `_create_task_dataset()` | Abstract on `BaseTrainer` | Required override — defines how a `GraphDataset` is mapped to a task-specific `DataLoader` |
| `evaluate()` | Abstract on `BaseTrainer` | Required override — defines evaluation protocol |
| `predict()` | Concrete on `SFTTrainer` only | Optional — generates top-k predictions with id→name mapping |
| `loss_functions` | Constructor of `SFTTrainer` | Strategy injection — any list of `SFTLoss(name, loss_fn, weight, target_node_type, is_distillation_loss)` |
| `TrainingArguments` | Constructor of `BaseTrainer` | All knobs (precision, eval strategy, split-graph, checkpointing) passed via dataclass |
| `GraphDatasetLoader` | Constructor of `BaseTrainer` | Swapable loader — any iterable yielding `GraphDataset(name, data)` |

### Distributed communication patterns

| Scenario | Mechanism | When |
|----------|-----------|------|
| Data-parallel forward | `DistributedDataParallel` | `world_size > 1` and NOT `split_graph_training` |
| Split-graph gradient sync | `dist.all_reduce(grad, op=AVG)` | `split_graph_training` and `world_size > 1` — manual AllReduce after `unscale_` |
| KGC ranking gather | `dist.all_reduce(ranking_tensor)`, `dist.all_reduce(cum_size)` | During `KGCTrainer.evaluate()` — cumulative ranking across shards |
| SFT prediction gather | `dist.all_gather_object(preds_list)` | During `SFTTrainer.predict()` — gathers per-rank prediction dictionaries |
| SFT eval gather | `utils.gather_results(rankings, targets)` | During `SFTTrainer.evaluate()` — unless split-graph mode (ranks already share data) |
| Logging gate | `if utils.get_rank() == 0` | `_log_metrics()`, `_save_checkpoint()`, console output |
