# gfmrag/text_emb_models/

## Responsibility

**Text embedding abstraction layer**. Provides a polymorphic interface (`BaseTextEmbModel.encode()`) for converting lists of text strings into fixed-dimensional `torch.Tensor` embeddings. This directory decouples downstream consumers (`GraphIndexDataset`, `GFMRetriever`, `Hipporag2Constructor`) from the choice of embedding backend (SentenceTransformer, custom HuggingFace NV-Embed-v2, or vLLM-hosted Qwen3), enabling backend substitution via Hydra config without code changes.

## Design Patterns

### Template Method
`BaseTextEmbModel` (in `base_model.py`) defines the **invariant skeleton** of embedding — batch-looping with progress bar, instruction prepending via `is_query` flag, optional L2 normalization, GPU→CPU transfer, and CUDA cache eviction. Subclasses override `encode()` to specialize the per-batch encoding call:
- `BaseTextEmbModel.encode()` — delegates to `SentenceTransformer.encode()`
- `NVEmbedV2.encode()` — calls `NVEmbedModel.encode()` (custom HF pipeline)
- `Qwen3TextEmbModel.encode()` — routes to `_make_request()` (OpenAI/vLLM API) or `embed()` (local `vLLM.LLM`)

### Strategy (via Hydra/OmegaConf)
The concrete embedding class is injected at runtime through `hydra.utils.instantiate()`:
```
# Config YAML
_target_: gfmrag.text_emb_models.Qwen3TextEmbModel
text_emb_model_name: Qwen/Qwen3-Embedding-0.6B
```
Consumers hold a `BaseTextEmbModel` reference. The strategy is selected by config path (e.g., `config/text_emb_model/qwen3.yaml`).

### Adapter
`NVEmbedV2` wraps a **custom HuggingFace architecture** (`NVEmbedModel` → `BidirectionalMistralModel` + `LatentAttentionModel`) behind the `BaseTextEmbModel.encode()` signature, normalizing `numpy.ndarray` returns to `torch.Tensor` and handling `NVEmbedModel`-specific arguments (`instruction`, `max_length`).

### Registry Pattern (HuggingFace Auto-Classes)
`modeling_nvembed.py` registers three model/configuration pairs into the HF `AutoModel` / `AutoConfig` registries at **module import time** (lines 522–530):
- `NVEmbedConfig` ↔ `NVEmbedModel` (model_type = `"nvembed"`)
- `LatentAttentionConfig` ↔ `LatentAttentionModel` (model_type = `"latent_attention"`)
- `BidirectionalMistralConfig` ↔ `BidirectionalMistralModel` (model_type = `"bidir_mistral"`)

This allows downstream serialization via `from_pretrained()` / `save_pretrained()`.

### Composition (NV-Embed-v2 Architecture)
`NVEmbedModel` composes:
1. A **backbone** `BidirectionalMistralModel` (causal→bidirectional by disabling `is_causal` on self-attention layers)
2. A **pooler** `LatentAttentionModel` (cross-attention to learned latent vectors + mean pooling with instruction masking)

### Constructor Parameter Object
Each concrete class accepts a flat set of constructor kwargs matching YAML config fields. `NVEmbedV2` normalizes `model_kwargs: Mapping` → `dict` at init.

## Abstractions & Interfaces

```
BaseTextEmbModel                    (abstract interface — though not formally ABC)
├── __init__(text_emb_model_name, normalize, batch_size, query_instruct, passage_instruct, model_kwargs)
└── encode(text: list[str], is_query: bool, show_progress_bar: bool) -> torch.Tensor
    ↑ overridden by:
    ├── BaseTextEmbModel.encode()       # SentenceTransformer backend
    ├── NVEmbedV2.encode()              # Custom NV-Embed HF backend (32768 ctx, right pad, EOS suffix)
    └── Qwen3TextEmbModel.encode()      # vLLM backend (local LLM or OpenAI-compatible API)
        ├── _make_request()             # REST API path (OpenAI client → vLLM server)
        └── embed()                     # Local vLLM process path (LLM.embed + PoolingParams)
```

Subclasses augment the interface with:
- `NVEmbedV2.add_eos(input_examples)` — appends `tokenizer.eos_token`
- `Qwen3TextEmbModel.add_instruct(instruct, query)` — prepends instruction string to query
- `Qwen3TextEmbModel._start_vllm_server()` — spawns `vllm.LLM` with distributed env cleanup
- `Qwen3TextEmbModel._is_api_available()` — health-check probe at `{api_base}/health`

## Data & Control Flow

### Entry → Exit
```
Caller (e.g., GraphIndexDataset)
  │
  │   text: list[str]          (raw strings)
  │   is_query: bool           (selects query_instruct vs passage_instruct)
  │
  ▼
encode()
  │
  ├── Empty guard: len(text)==0 → torch.empty((0,0))
  │
  ├── Select instruction prompt
  │     query_instruct if is_query else passage_instruct
  │
  ├── Per-concrete-class dispatch:
  │
  │   A) BaseTextEmbModel
  │      └─ SentenceTransformer.encode(batch, prompt=prompt, normalize_embeddings=normalize,
  │                                     device="cuda"|"cpu", convert_to_tensor=True).float()
  │
  │   B) NVEmbedV2
  │      └─ NVEmbedModel.encode(batch, instruction=prompt, max_length=32768)
  │         ├─ input_transform_func()        → tokenizer(PREFIX + text + EOS)
  │         ├─ prepare_kwargs_from_batch()   → instruction masking on attention_mask
  │         ├─ NVEmbedModel.forward()        → BidirectionalMistralModel → LatentAttentionModel
  │         └─ returns np.ndarray | torch.Tensor
  │
  │   C) Qwen3TextEmbModel
  │      ├─ add_instruct() per string         → prepends "Instruct: ...\nQuery: "
  │      ├─ if api_base is set:
  │      │   └─ _make_request()               → OpenAI client.embeddings.create() REST call
  │      └─ else:
  │          └─ embed()                       → vLLM LLM.embed(batch, PoolingParams)
  │
  ├── Optional L2 normalization (NVEmbedV2, Qwen3 via normalize flag)
  ├── .cpu() transfer
  ├── del batch_embeddings + torch.cuda.empty_cache()
  │
  ▼
  torch.Tensor               (all batches concatenated on dim=0, on CPU)
  shape: (len(text), embedding_dim)
```

### Qwen3 Dual-Mode Startup

```
Qwen3TextEmbModel.__init__()
  │
  ├── api_base is None?
  │     → _start_vllm_server()
  │        1. Save & clear dist env vars (RANK, WORLD_SIZE, MASTER_ADDR, etc.)
  │        2. Set CUDA_VISIBLE_DEVICES = old LOCAL_RANK (default "0")
  │        3. Set VLLM_WORKER_MULTIPROC_METHOD = "spawn"
  │        4. LLM(model, enforce_eager=True, task="embed", hf_overrides={"is_matryoshka": True})
  │        5. Restore dist env vars
  │
  └─ api_base is set?
        → _is_api_available()  GET {api_base}/health (5s timeout)
            ├── 200 → store OpenAI client
            └── not 200 → raise RuntimeError
```

## Integration Points

### Consumers (import `BaseTextEmbModel`)

| Consumer Module | File | Usage |
|---|---|---|
| `GraphIndexDataset` | `graph_index_datasets/graph_index_dataset.py` | `instantiate(self.text_emb_model_cfgs)` in `process_graph()` (line 459) and `process_text()` (line 629) |
| `GraphIndexDatasetV1` | `graph_index_datasets/graph_index_dataset_v1.py` | `instantiate(self.text_emb_model_cfgs)` in `process()` (line 238) |
| `GFMRetriever` | `gfmrag_retriever.py` | Constructor parameter (line 54); forwarded from `from_index()` which uses `instantiate(qa_data.text_emb_model_cfgs)` (line 292) |
| `Hipporag2Constructor` | `graph_index_construction/sft_constructors/hipporag2_constructor.py` | Constructor parameter (line 53) |
| `DPRELModel` / `NVEmbedV2ELModel` | `graph_index_construction/entity_linking_model/dpr_el_model.py` | Specialization of entity linking that reuses NV-Embed-v2 architecture (line 169) |

### Configuration Entrypoints

Hydra config files under `gfmrag/workflow/config/text_emb_model/`:

| Config | `_target_` | Model |
|---|---|---|
| `gte_qwen2_1.5b.yaml` | `gfmrag.text_emb_models.BaseTextEmbModel` | Alibaba-NLP/gte-Qwen2-1.5B-instruct |
| `gte_qwen2_7b.yaml` | `gfmrag.text_emb_models.BaseTextEmbModel` | Alibaba-NLP/gte-Qwen2-7B-instruct |
| `qwen3.yaml` | `gfmrag.text_emb_models.Qwen3TextEmbModel` | Qwen/Qwen3-Embedding-0.6B |
| `qwen3_8b.yaml` | `gfmrag.text_emb_models.Qwen3TextEmbModel` | Qwen/Qwen3-Embedding-8B |

Referenced via OmegaConf in:
- `gfmrag/gfmrag_retriever.py` — `dataset_kwargs["text_emb_model_cfgs"] = OmegaConf.create(...)` (line 196)
- `gfmrag/utils/util.py` — `cfg.datasets.cfgs.text_emb_model_cfgs` (line 22)
- `gfmrag/workflow/experiments/visualize_path.py` — `OmegaConf.create(model_config["text_emb_model_config"])` (line 49)

### Dependencies

| Dependency | Used by | Purpose |
|---|---|---|
| `sentence-transformers` | `BaseTextEmbModel` | Embedding backend via `SentenceTransformer` |
| `transformers` (AutoModel, AutoConfig, MistralModel, PreTrainedModel) | `NVEmbedV2`, `modeling_nvembed`, `configuration_nvembed` | Custom HuggingFace model definition, registration, and serialization |
| `vllm` | `Qwen3TextEmbModel` (local mode) | `LLM.embed()` for inference-optimized embedding |
| `openai` | `Qwen3TextEmbModel` (API mode) | `OpenAI().embeddings.create()` REST client |
| `datasets` | `modeling_nvembed` | `Dataset.from_dict()` for batch encoding DataLoader |
| `einops` | `modeling_nvembed` | `rearrange`/`repeat` for attention head manipulation |
| `hydra` / `omegaconf` | Consumers | `instantiate()` constructs concrete model from config |

### Public API Surface (`__init__.py`)

```python
__all__ = ["BaseTextEmbModel", "NVEmbedV2", "Qwen3TextEmbModel"]
```

All three classes are re-exported at the package level for `instantiate()` resolution via dotted path `gfmrag.text_emb_models.<ClassName>`.

### HuggingFace Registry Hooks (side effects at import)

`configuration_nvembed.py` lines 86–88:
```python
AutoConfig.register("nvembed", NVEmbedConfig)
AutoConfig.register("latent_attention", LatentAttentionConfig)
AutoConfig.register("bidir_mistral", BidirectionalMistralConfig)
```

`modeling_nvembed.py` lines 522–529:
```python
AutoModel.register(NVEmbedConfig, NVEmbedModel)
AutoModel.register(LatentAttentionConfig, LatentAttentionModel)
AutoModel.register(BidirectionalMistralConfig, BidirectionalMistralModel)
# Also registers for auto-class
NVEmbedModel.register_for_auto_class("AutoModel")
LatentAttentionModel.register_for_auto_class("AutoModel")
BidirectionalMistralModel.register_for_auto_class("AutoModel")
```

These registrations enable `NVEmbedModel.from_pretrained()` for NV-Embed-v2 checkpoints and allow the NV-Embed custom architecture to be loaded via `AutoModel.from_pretrained()`.
