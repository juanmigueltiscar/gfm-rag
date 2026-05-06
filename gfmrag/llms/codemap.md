# `gfmrag/llms/` — Language Model Abstraction Layer

## Responsibility

This module defines a **Strategy** abstraction over heterogeneous LLM backends (local Hugging Face models, OpenAI, Google Gemini, Mistral AI). It provides a uniform interface (`BaseLanguageModel`) that the rest of the system uses for text generation and token-length estimation, decoupling downstream workflows from any specific model provider or deployment topology.

The module is **not** a model-training or fine-tuning package; it is strictly an **inference adapter layer**.

---

## Design Patterns

### Strategy (primary)

- **Abstract interface**: `BaseLanguageModel` (ABC) in `base_language_model.py` declares two abstract methods:
  - `token_len(text: str) -> int`
  - `generate_sentence(llm_input: str | list, system_input: str = "") -> str | Exception`
- **Concrete strategies**:
  | Class | File | Backend |
  |---|---|---|
  | `HfCausalModel` | `base_hf_causal_model.py` | Hugging Face `transformers` (local inference, supports quantization) |
  | `ChatGPT` | `chatgpt.py` | OpenAI Chat Completions API |
  | `Gemini` | `gemini.py` | Google `generativeai` API |
  | `MistralSmall32API` | `mistral.py` | Mistral AI native `mistralai` client |
- **Selection mechanism**: The concrete class is chosen at **instantiation time** by the caller (usually via Hydra `instantiate` in workflow configs). There is no dynamic factory or registry — the `__init__.py` explicitly exports `BaseLanguageModel`, `HfCausalModel`, and `ChatGPT`; `Gemini` and `MistralSmall32API` are importable but **not** re-exported.

### Template Method (implicit)

Each concrete strategy follows the same internal template in `generate_sentence()`:
1. Normalise `llm_input` (string or `list[dict]`) into a uniform message structure.
2. (Optional) Truncate if token count exceeds `maximun_token`.
3. Retry loop with configurable retry count and backoff.
4. Invoke backend-specific API, return generated text or the final `Exception`.

This structure is replicated across all four implementations (no shared base-class orchestration), so it is an **ad-hoc Template Method** rather than a formal one enforced by the ABC.

### Error-as-Value Return

`generate_sentence()` returns `str | Exception` — errors are **not** raised but returned as values. Callers are expected to check `isinstance(response, Exception)` before consuming the result. This is a deliberate design choice to make retry-or-fail logic explicit at the call site.

---

## Data & Control Flow

### Entry

1. A consumer (e.g., `qa_ircot_inference.py` or `rerank.py`) obtains a concrete `BaseLanguageModel` instance.
   - In workflows: instantiated via Hydra `instantiate(cfg.llm)` from YAML config.
   - In `rerank.py`: directly constructed as `ChatGPT(model_name_or_path=..., retry=...)`.

2. The consumer calls `llm.generate_sentence(prompt, system_input="")`.
   - `prompt` is either a plain `str` or a `list[dict]` with `{"role": ..., "content": ...}` keys.

### Within `generate_sentence()`

The method branches on `isinstance(llm_input, list)` to decide message format:

- **String input**: wraps into `[{"role": "user", "content": llm_input}]`, optionally prepending `{"role": "system", "content": system_input}`.
- **List input**: used as-is (assumed to be a fully-formed chat message list).

Then each implementation sends the message to its backend:

| Strategy | Backend call | Token counting |
|---|---|---|
| `HfCausalModel` | `self.generator(message, ...)` (HF text-generation pipeline) | `self.tokenizer.tokenize(text)` |
| `ChatGPT` | `self.client.chat.completions.create(model=..., messages=message, ...)` | `tiktoken.encoding_for_model(self.model_name)` |
| `Gemini` | `self.client.generate_content(contents=..., generation_config=...)` | `self.client.count_tokens(text)` |
| `MistralSmall32API` | `self.client.chat(model=..., messages=messages, ...)` | `tiktoken.encoding_for_model("gpt-4")` (approximation) |

### Error & Retry State Machine

```
[call generate_sentence] ──► retry_count = 0
       │
       ▼
 ┌─ try API call ──► success ──► return str
 │       │
 │       ▼
 │     failure (any Exception)
 │       │
 │       ▼
 │   log error, sleep(backoff), retry_count++
 │       │
 │     retry_count <= retry? ──► yes ──► retry
 │       │
 │       no
 │       ▼
 └──► return Exception (last error)
```

- Backoff: `ChatGPT` and `Gemini` use fixed 30 s sleep; `MistralSmall32API` uses exponential backoff `15 * (2^cur_retry)`.
- `HfCausalModel` has **no retry loop** — it calls the pipeline once and returns either the generated text or the exception.

### Exit

- **Success**: `str` — generated text, stripped of leading/trailing whitespace.
- **Failure**: `Exception` instance (the last caught exception, or `Exception("Failed to generate sentence")` if the initial call raised).

---

## Integration Points

### Public API (exposed via `__init__.py`)

```python
__all__ = ["BaseLanguageModel", "HfCausalModel", "ChatGPT"]
```

`Gemini` and `MistralSmall32API` are importable (e.g., `from gfmrag.llms.gemini import Gemini`) but **not** part of the public surface — this is likely an oversight or work-in-progress.

### Consumers

| Consumer | File | Usage |
|---|---|---|
| IRCoT inference pipeline | `gfmrag/workflow/qa_ircot_inference.py` | Imports `BaseLanguageModel` (type annotation), receives instance via Hydra `instantiate` |
| HiPPo-RAG2 reranking | `gfmrag/graph_index_construction/sft_constructors/hipporag2/rerank.py` | Imports `ChatGPT`, constructs directly with `ChatGPT(model_name_or_path=..., retry=...)` |

### Environment Variables (loaded via `python-dotenv`)

| Variable | Used by | Purpose |
|---|---|---|
| `HF_TOKEN` | `HfCausalModel` | Hugging Face authentication for gated models |
| `OPENAI_API_KEY` | `ChatGPT` | OpenAI API authentication (read by `openai.OpenAI()` internally) |
| `GOOGLE_API_KEY` | `Gemini` | Google AI API authentication |
| `MISTRAL_API_KEY` | `MistralSmall32API` | Mistral AI API authentication |

### Framework Dependencies

| Dependency | Used by | Purpose |
|---|---|---|
| `transformers` | `HfCausalModel` | Model loading, tokenization, text-generation pipeline |
| `torch` | `HfCausalModel` | `torch.inference_mode()`, dtype selection, device_map |
| `openai` | `ChatGPT` | OpenAI REST client |
| `google.generativeai` | `Gemini` | Google Generative AI SDK |
| `mistralai` | `MistralSmall32API` | Mistral AI native Python client |
| `tiktoken` | `ChatGPT`, `MistralSmall32API` | Token counting (OpenAI BPE tokeniser) |

### Configuration Surface

Each concrete strategy accepts constructor arguments. When used via Hydra, these are populated from model-specific YAML configs:

| Parameter | Applies to | Description |
|---|---|---|
| `model_name_or_path` | All | Model identifier (HF hub name, OpenAI model ID, Gemini model name, Mistral model ID) |
| `maximun_token` | `HfCausalModel`, `ChatGPT`, `Gemini`, `MistralSmall32API` | Input token cap |
| `max_new_tokens` | `HfCausalModel` | Output token limit |
| `dtype` | `HfCausalModel` | One of `fp32`, `fp16`, `bf16` |
| `quant` | `HfCausalModel` | `None`, `"4bit"`, or `"8bit"` |
| `attn_implementation` | `HfCausalModel` | `"eager"`, `"sdpa"`, or `"flash_attention_2"` |
| `retry` | `ChatGPT`, `Gemini`, `MistralSmall32API` | Max retry attempts for API calls |

### Notable Caveats

1. **`__init__.py` is stale**: `Gemini` and `MistralSmall32API` are not re-exported, so `from gfmrag.llms import Gemini` will fail. Consumers must use deep imports.
2. **`HfCausalModel` has no retry loop**: unlike the API-backed strategies, local inference errors propagate immediately.
3. **Token truncation logic is unreliable**: `ChatGPT` and `Gemini` attempt to right-truncate `llm_input` by index-slicing, but `llm_input` may be a `list[dict]` — slicing a list of messages is semantically wrong and will produce malformed message arrays.
4. **Docstrings mislabel providers**: `Gemini` and `MistralSmall32API` docstrings still refer to "ChatGPT" (copy-paste artifacts).
5. **`maximun_token` typo**: the attribute is consistently misspelled (missing 'm') across all implementations.
