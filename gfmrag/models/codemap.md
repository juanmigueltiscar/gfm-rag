# `gfmrag/models/` — Model Architectures

## Responsibility

Defines the graph neural network (GNN) model architectures used for knowledge-graph reasoning and document retrieval within the GFM-RAG framework. This package provides:

- An **abstract GNN interface** (`BaseGNNModel`) that all concrete models implement.
- **Two model families**: GFM-RAG-v1 (retrieval-oriented) and GFM-Reasoner (complex-query reasoning).
- The **ULTRA NBFNet** (Neural Bellman-Ford Network) message-passing backbone, which is a third-party module excluded from linting.
- A **document ranker strategy hierarchy** for mapping entity-level predictions to document scores.
- A **fused message-passing kernel** (`rspmm`) — a custom CUDA/C++ extension for efficient graph convolution.

The package is the "neural engine" of the GFM-RAG pipeline, transforming graph-structured knowledge and query embeddings into node-level scores.

---

## Design Patterns

### 1. Template Method

**`BaseGNNModel`** (`base_model.py`) defines the abstract contract:

```python
class BaseGNNModel(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor: ...
```

Two concrete subclasses implement the template:
| Package | Class | Role |
|---|---|---|
| `gfm_rag_v1` | `QueryGNN` | Retrieval-oriented query embedding + entity scoring |
| `gfm_reasoner` | `QueryGNN` | Complex-query reasoning with entity embedding fusion |

### 2. Inheritance Chain (NBFNet)

```
BaseNBFNet  (ultra/base_nbfnet.py)
    └── EntityNBFNet  (ultra/models.py)
            └── QueryNBFNet  (ultra/models.py)
```

- **`BaseNBFNet`** provides the canonical Bellman-Ford message-passing loop, edge dropping via `remove_easy_edges()`, negative-sample-to-tail conversion, beam-search path extraction (`beam_search_distance`, `topk_average_length`), and visualization hooks.
- **`EntityNBFNet`** layers `GeneralizedRelationalConv` modules, adds a scoring MLP, and implements `forward()` for standard KGC (knowledge graph completion) with triple batches.
- **`QueryNBFNet`** overrides `bellmanford()` to accept pre-computed `node_features` and `query` tensors (instead of reading from triple batches), enabling it to be used as a sub-module in retrieval/reasoning pipelines. Also adds distributed inference paths (METIS partition and split-graph AllGather strategies).

### 3. Strategy (Ranker)

**`BaseDocRanker`** (`gfm_rag_v1/rankers.py`) is a strategy interface:

```python
class BaseDocRanker(ABC):
    @abstractmethod
    def __call__(self, ent_pred: Tensor, ent2doc: Tensor) -> Tensor: ...
```

Concrete strategies injected into `GNNRetriever` via constructor:

| Ranker | Behavior |
|---|---|
| `SimpleRanker` | `ent_pred @ ent2doc` (sparse matmul) |
| `IDFWeightedRanker` | IDF-weight entity scores before matmul |
| `TopKRanker` | Binarize top-K entity scores, then matmul |
| `IDFWeightedTopKRanker` | IDF-weight only top-K entities, then matmul |

This is genuine **dependency injection** — `GNNRetriever.__init__` receives `ranker: BaseDocRanker` and calls it in `map_entities_to_docs()`.

### 4. Factory Dispatch (RSPMM)

`generalized_rspmm()` in `rspmm/rspmm.py` uses a **dynamic function lookup** pattern:

```python
name = f"RSPMM{sum.capitalize()}{mul.capitalize()}Function"
Function = getattr(module, name)
```

It dispatches to one of 6 `autograd.Function` classes based on the `sum` and `mul` arguments (e.g., `RSPMMAddMulFunction`, `RSPMMMaxAddFunction`). Each bridges to a compiled CUDA/C++ kernel.

### 5. Composition

Both `gfm_rag_v1.QueryGNN` and `gfm_reasoner.QueryGNN` compose with an `entity_model` (either `EntityNBFNet` or `QueryNBFNet`) passed at construction time. The outer model produces query-dependent node features; the inner model performs message-passing over the graph.

### 6. Logical Query Encoding (ULTRA `query_utils.py`)

The `Query` class (subclass of `torch.Tensor`) encodes complex logical queries (projection, intersection, union, negation) in **postfix notation** with bitmask operators. Supports conversion from nested tuples (BetaE format), human-readable strings, and computation-graph analysis.

---

## Data & Control Flow

### Flow 1: GFM-RAG v1 Retrieval (`GNNRetriever`)

```
Input:  graph (Data), batch dict with "question_embeddings" + "start_nodes_mask"
        │
        ├─ question_mlp(question_emb)        # (bs, emb_dim)
        ├─ rel_mlp(graph.rel_attr)            # (num_rels, emb_dim) → expanded to (bs, num_rels, emb_dim)
        │
        ├─ Optional: entities_weight via get_entities_weight()
        │     (inverse frequency of entity→doc mapping)
        │
        ├─ einsum("bn,bd->bnd", mask, query_emb)
        │     → initial node features (boundary condition)
        │
        ├─ entity_model(graph, node_feat, rel_repr, query_emb)
        │     └─ QueryNBFNet.bellmanford()
        │           ├─ For each GeneralizedRelationalConv layer:
        │           │     ├─ Message: distmult/transe/rotate on (input_j, relation_j)
        │           │     ├─ Aggregate: sum/mean/max/min/PNA via rspmm kernel
        │           │     ├─ Update: linear(cat([input, update]))
        │           │     └─ Residual: optional short_cut
        │           └─ Concat hidden states + node_query → output (bs, num_nodes, feat_dim)
        │     └─ mlp(output) → score (bs, num_nodes)
        │
        ├─ map_entities_to_docs(score, graph)
        │     └─ For each target type: ranker(ent_pred, ent2doc_mapping)
        │           └─ Sparse matmul → doc scores
        │
Output: score (bs, num_docs) — ranking scores for document nodes
```

### Flow 2: GFM-Reasoner (`GraphReasoner`)

```
Input:  graph (Data), batch dict with "question_embeddings" + "start_nodes_mask"
        │
        ├─ question_mlp, rel_mlp (same as Flow 1)
        │
        ├─ Optional entity weight multiplication
        │
        ├─ einsum → node_embedding
        │
        ├─ Entity embedding fusion (configurable):
        │     None              — no entity embeddings
        │     "early-fusion"    — ent_mlp(graph.x) → cat + early_fuse_mlp
        │     "late-fusion"     — entity embeddings concatenated after GNN
        │     "early-late-fusion" — both fusions
        │
        ├─ entity_model(graph, node_feat, rel_repr, query_emb) → output (bs, num_nodes, emb_dim)
        │
        ├─ Late-fusion MLP (if configured):
        │     predict_mlp(cat([output, ent_emb])) → (bs, num_nodes)
        │
Output: score (bs, num_nodes) — entity-level scoring distribution
```

### Flow 3: Knowledge Graph Completion Pretraining (`EntityNBFNet`)

```
Input:  data (Data), batch (bs, 1+num_neg, 3) — (head, tail, relation) triples
        │
        ├─ remove_easy_edges(data, h, t, r)  — dynamic edge dropout for training
        ├─ negative_sample_to_tail()          — convert to tail-prediction mode
        │
        ├─ bellmanford(data, h_idx, r_idx)
        │     └─ (same layer iteration as above)
        │
        ├─ gather tail representations from output
        ├─ mlp(feature) → score
        │
Output: score (bs, 1+num_neg) — logits for positive + negative triples
```

### State Transitions (Visualization Path)

`GNNRetriever.visualize()` / `GraphReasoner.visualize()` / `QueryNBFNet.visualize()`:

1. Runs `bellmanford()` with `separate_grad=True` (edge weights require grad).
2. Extracts score for a target entity via `mlp()`.
3. Backpropagates to get `edge_grads = autograd.grad(score, edge_weights)`.
4. Runs `beam_search_distance()` to find top-K paths from query entities to target.
5. Backtracks via `topk_average_length()` to produce human-readable reasoning paths.

---

## Module Map

| File | Exports | Dependencies |
|---|---|---|
| `__init__.py` | `BaseGNNModel` | (re-export from `base_model.py`) |
| `base_model.py` | `BaseGNNModel` | `torch.nn.Module`, `abc` |
| `gfm_rag_v1/__init__.py` | `QueryGNN`, `GNNRetriever` | — |
| `gfm_rag_v1/model.py` | `QueryGNN`, `GNNRetriever` | `BaseGNNModel`, `EntityNBFNet`, `QueryNBFNet`, `BaseDocRanker` |
| `gfm_rag_v1/rankers.py` | `BaseDocRanker`, `SimpleRanker`, `IDFWeightedRanker`, `TopKRanker`, `IDFWeightedTopKRanker` | `torch` |
| `gfm_reasoner/__init__.py` | `QueryGNN`, `GraphReasoner` | — |
| `gfm_reasoner/model.py` | `QueryGNN`, `GraphReasoner` | `BaseGNNModel`, `QueryNBFNet` |
| `ultra/__init__.py` | `EntityNBFNet`, `QueryNBFNet` | — |
| `ultra/base_nbfnet.py` | `BaseNBFNet` + helpers | `variadic`, `tasks` |
| `ultra/models.py` | `EntityNBFNet`, `QueryNBFNet` | `BaseNBFNet`, `layers` |
| `ultra/layers.py` | `GeneralizedRelationalConv` | `MessagePassing`, `variadic`, `rspmm` |
| `ultra/tasks.py` | `edge_match`, `negative_sampling`, `build_relation_graph`, etc. | `torch`, `Data` |
| `ultra/query_utils.py` | `Query`, `Stack`, `evaluate`, `batch_evaluate`, etc. | `torch.distributed`, `variadic` |
| `ultra/variadic.py` | `native_scatter`, variadic ops | `torch` |
| `ultra/util.py` | Config loading, logging, dist helpers | `jinja2`, `yaml`, `easydict` |
| `ultra/rspmm/__init__.py` | `generalized_rspmm` | — |
| `ultra/rspmm/rspmm.py` | 6 `autograd.Function` classes, `load_extension` | C++/CUDA sources in `source/` |

---

## Integration Points

### Consumers (Import Graph)

| Consumer Module | What It Imports | Purpose |
|---|---|---|
| `gfmrag.gfmrag_retriever` | `BaseGNNModel`, `EntityNBFNet`, `QueryNBFNet`, `query_utils` | `GFMRetriever` constructs and runs the retrieval model |
| `gfmrag.trainers.kgc_trainer` | `tasks` | KGC pretraining: negative sampling, evaluation |
| `gfmrag.trainers.sft_trainer` | `query_utils` | SFT training: query encoding, distributed helpers |
| `gfmrag.losses` | `variadic_softmax` | Loss function computation |
| `gfmrag.utils.qa_utils` | `variadic` | QA evaluation utilities |
| `gfmrag.workflow.qa` | `query_utils` | QA pipeline |
| `gfmrag.workflow.qa_ircot_inference` | `query_utils` | IRCoT inference |
| `gfmrag.workflow.experiments.visualize_path` | `query_utils` | Path visualization |

### Exported API Surface

```
gfmrag.models
    └── BaseGNNModel            — Abstract model base class

gfmrag.models.gfm_rag_v1
    ├── QueryGNN                — Query-dependent entity scorer (KGC-style)
    └── GNNRetriever            — End-to-end document retriever (extends QueryGNN)

gfmrag.models.gfm_reasoner
    ├── QueryGNN                — Query-dependent entity scorer (reasoner variant)
    └── GraphReasoner           — Entity-level reasoner (extends QueryGNN)

gfmrag.models.ultra
    ├── EntityNBFNet            — NBFNet for KGC entity prediction
    ├── QueryNBFNet             — NBFNet for query-dependent reasoning
    ├── query_utils             — Query encoding, evaluation, distributed helpers
    ├── tasks                   — Negative sampling, relation graph builder
    ├── variadic                — Variadic tensor operations (scatter, sort, softmax)
    └── rspmm                   — Fused message-passing GPU kernel
```

### Key Hooks and Extension Points

1. **`BaseDocRanker`** — Strategy interface for entity→document mapping; injectable into `GNNRetriever`
2. **`BaseGNNModel.forward()`** — Template method; all models must implement
3. **`GeneralizedRelationalConv.message_func`** — Switch between `"transe"`, `"distmult"`, `"rotate"`
4. **`GeneralizedRelationalConv.aggregate_func`** — Switch between `"sum"`, `"mean"`, `"max"`, `"pna"`, `"min"`
5. **`BaseNBFNet.remove_one_hop`** — Toggle dynamic edge removal during training
6. **`BaseNBFNet.concat_hidden`** — Whether to concatenate all layer outputs or use only the last
7. **`gfm_reasoner.QueryGNN.use_ent_emb`** — Entity embedding fusion strategy (`None`, `"early-fusion"`, `"late-fusion"`, `"early-late-fusion"`)
8. **`BaseNBFNet.beam_search_distance()` + `topk_average_length()`** — Path extraction for model interpretability

### External Dependencies

| Dependency | Purpose |
|---|---|
| `torch` | Core tensor ops, neural network modules, autograd |
| `torch_geometric` | `Data` class, `MessagePassing` base class |
| `torch.distributed` | Multi-GPU inference (AllGather, barrier) |
| C++14 + CUDA (build-time) | `rspmm` custom kernel compilation via `torch.utils.cpp_extension` |

---

## Notes

- The `ultra/` subpackage is **third-party code excluded from ruff linting** (per `AGENTS.md`). It should not be reformatted.
- All `ultra/` Python files carry `# mypy: ignore-errors` to suppress type-checking.
- **Inverse relations are created automatically** during data processing (handled in `GraphIndexDataset.process_graph()`, not in this module). The model relies on `num_relations` being even (direct + inverse pairs).
- The `rspmm` CUDA kernel requires sorted edge indices (asserted in each autograd Function) and supports exactly the message functions "add" (TransE) and "mul" (DistMult) for fused execution. Other message functions (`rotate`) fall back to separate message+aggregate.
- Distributed inference in `QueryNBFNet.bellmanford()` supports two strategies:
  - **Edge-partition** (`partition_graph_edges`): each rank owns a slice of target nodes; full source states are AllGathered before each layer.
  - **METIS partition** (`partition_graph_metis`): only boundary node states are communicated (scatter → AllGather).
