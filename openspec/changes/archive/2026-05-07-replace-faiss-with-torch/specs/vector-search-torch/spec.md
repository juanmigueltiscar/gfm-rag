## ADDED Requirements

### Requirement: L2 normalization of embeddings

The system SHALL normalize embedding vectors to unit L2 norm using `torch.nn.functional.normalize` with `p=2` and `dim=1`, producing equivalent results to `faiss.normalize_L2`.

#### Scenario: Normalize a batch of embeddings
- **WHEN** a batch of embeddings with shape `(N, D)` is passed to `_encode_texts`
- **THEN** the returned tensor SHALL have L2 norm of 1.0 for each row within `atol=1e-5`

#### Scenario: Normalize empty batch
- **WHEN** an empty text list is passed to `_encode_texts`
- **THEN** the returned tensor SHALL have shape `(0, 0)` and dtype `float32`

### Requirement: Flat inner-product index construction

The system SHALL store embedding vectors as a `torch.Tensor` of shape `(N, D)` with dtype `float32`, replacing `faiss.IndexFlatIP`. The index tensor SHALL be kept on CPU to match original FAISS behavior.

#### Scenario: Build index from embeddings
- **WHEN** `_build_faiss_index` is called with a non-empty embeddings array of shape `(N, D)`
- **THEN** the method SHALL return a `torch.Tensor` of shape `(N, D)` and dtype `float32`

#### Scenario: Build index from empty embeddings
- **WHEN** `_build_faiss_index` is called with an empty array (size 0)
- **THEN** the method SHALL return `None`

### Requirement: Top-K inner product search

The system SHALL compute inner-product similarity between a query vector and all index vectors via `torch.mm(query, index.T)`, then return the top-K indices and scores using `torch.topk(scores, k, largest=True)`.

#### Scenario: Search returns valid results
- **WHEN** a query embedding of shape `(1, D)` is passed to `_search_by_type` with `top_k > 0`
- **THEN** the method SHALL return at most `top_k` results, each with a label from `nodes_by_type[node_type]` and a normalized score in `[0, 1]`

#### Scenario: Search with index too small
- **WHEN** `top_k` exceeds the number of vectors in the index (`ntotal`)
- **THEN** the method SHALL return `min(top_k, ntotal)` results

#### Scenario: Search with no index
- **WHEN** the node type has no corresponding index (`None`)
- **THEN** the method SHALL return empty lists

### Requirement: Torch version supports aarch64 CUDA

The project SHALL depend on `torch>=2.9.0` to enable CUDA acceleration on `aarch64` Linux platforms through the `nvidia-*` package family.

#### Scenario: torch resolves with CUDA on aarch64
- **WHEN** `uv sync` is run on an aarch64 Linux machine with CUDA toolkit installed
- **THEN** torch SHALL be installed with CUDA support (`torch.cuda.is_available()` returns `True`)

### Requirement: FAISS dependency removed

The project SHALL NOT depend on `faiss-gpu-cu12` or any other `faiss` package.

#### Scenario: uv lock resolves without FAISS
- **WHEN** `uv lock` is run
- **THEN** the lockfile SHALL contain no `faiss-gpu-cu12` or `faiss-cpu` entries
