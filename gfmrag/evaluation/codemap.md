# gfmrag/evaluation/

## Responsibility

**Benchmark-specific evaluator strategies** for multi-hop QA and retrieval tasks. This package provides concrete scoring implementations for four distinct QA benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA) plus a standalone retrieval recall metric. Each evaluator ingests a JSONL prediction file produced by an upstream inference pipeline and returns a dictionary of aggregated metrics (exact match, F1 score, precision, recall, and/or recall@k). The package has zero non-standard-library dependencies — it relies solely on `json`, `abc`, `re`, `string`, `collections`, `statistics`, and its own base class.

## Design Patterns

### 1. Template Method (abstract base class)

`BaseEvaluator` (in `base_evaluator.py`) defines the invariant skeleton of the evaluation algorithm:

- **Common step** (implemented): `__init__(prediction_file: str)` opens the JSONL file, deserialises each line into a `data: list[dict]`.
- **Custom step** (abstract): `evaluate() -> dict` is overridden by every subclass to implement benchmark-specific metric logic.

All four concrete evaluators inherit `BaseEvaluator` and differ only in their `evaluate()` implementation.

### 2. Strategy (polymorphic evaluator selection)

Each concrete subclass represents a strategy for computing metrics over a different data schema:

| Concrete Strategy | Input Keys Expected | Output Metrics |
|---|---|---|
| `RetrievalEvaluator` | `supporting_documents`, `retrieved_docs` | `recall@{1,2,5,10}` |
| `HotpotQAEvaluator` | `response`, `answer` | `em`, `f1`, `precision`, `recall` |
| `MusiqueEvaluator` | `response`, `answer`, `answer_aliases` | `em`, `f1`, `precision`, `recall` |
| `TwoWikiQAEvaluator` | `response`, `answer`, `answer_aliases` | `em`, `f1`, `precision`, `recall` |

Consumers instantiate the desired evaluator by class name via the public `__init__.py` exports. No factory or registry exists — selection is static at the call site.

### 3. Pure-function utility layer

Module-level pure functions (no side effects, no instance state) encapsulate text normalisation and pairwise metric computation:

- `normalize_answer(s: str) -> str` — lowercases, strips punctuation/articles, normalises whitespace. Defined independently and identically in `hotpot_qa_evaluator.py`, `musique_evaluator.py`, and `two_wiki_qa_evaluator.py` (noticeable **code duplication** — the three QA evaluators each copy the same normalisation pipeline).
- `f1_score(prediction, ground_truth) -> tuple[float, float, float]` — returns `(f1, precision, recall)`.
- `exact_match_score(prediction, ground_truth) -> int | bool` — 1/True if normalised strings match.
- `update_answer(metrics, prediction, gold) -> tuple` — mutates an in-flight metrics dict and returns the per-example scores.

### 4. Multi-ground-truth aggregation (MusiqueEvaluator, TwoWikiQAEvaluator)

Both MuSiQue and 2WikiMultiHopQA provide answer aliases (`answer_aliases: list[str]`). These evaluators implement a **max-over-ground-truths** reduction:

- `metric_max_over_ground_truths(metric_fn, prediction, ground_truths)` — returns the best score across all aliases for scalar metrics.
- `metric_max_f1_over_ground_truths(metric_fn, prediction, ground_truths)` — returns the F1 tuple from the alias that maximises F1.

`HotpotQAEvaluator` does *not* use multi-ground-truth aggregation — it compares only against `pred["answer"]` directly.

## Data & Control Flow

### Entry point

Every evaluator is instantiated with a single constructor argument:

```
evaluator = HotpotQAEvaluator(prediction_file="results/predictions.jsonl")
```

### Input contract

File is a **JSONL** file (one JSON object per line). Minimal required keys per concrete evaluator:

| Evaluator | Required keys in each JSON line |
|---|---|
| `RetrievalEvaluator` | `supporting_documents: list[str]`, `retrieved_docs: dict[str, list[{"id": str, "score": float}]]` |
| `HotpotQAEvaluator` | `response: str`, `answer: str` |
| `MusiqueEvaluator` | `response: str`, `answer: str`, `answer_aliases: list[str]` |
| `TwoWikiQAEvaluator` | `response: str`, `answer: str`, `answer_aliases: list[str]` |

### State transition

```
BaseEvaluator.__init__(prediction_file)
    │
    ├── open(prediction_file, "r")
    ├── [json.loads(line) for line in f]           → self.data: list[dict]
    │
    └── (ready to evaluate)

evaluator.evaluate()
    │
    ├── [for each pred in self.data]               ← iterate over all predictions
    │   │
    │   ├── Extract predicted answer               ← "Answer: ..." suffix split
    │   ├── Extract gold answer(s)                 ← single answer or answer + aliases
    │   ├── Compute per-example metric             ← exact_match, f1_score, or recall@k
    │   └── Accumulate into metrics dict
    │
    ├── Normalise by n = len(self.data)             ← macro-average over all examples
    │
    └── Return metrics: dict[str, float]
```

### Output contract

All QA evaluators return a dictionary of four macro-averaged metrics:
```python
{"em": 0.723, "f1": 0.814, "precision": 0.852, "recall": 0.791}
```

`RetrievalEvaluator` returns `k` entries keyed as `recall@{k}`:
```python
{"recall@1": 0.45, "recall@2": 0.62, "recall@5": 0.83, "recall@10": 0.91}
```

### Retrieval-specific flow

`RetrievalEvaluator.evaluate(k=(1, 2, 5, 10))` differs from the QA evaluators:

1. It accepts an optional `k` tuple parameter (default `(1, 2, 5, 10)`).
2. It flattens `pred["retrieved_docs"]["<hop_id>"]` lists into a single list of `{"id": str, "score": float}`.
3. It sorts by `score` descending, extracts `id` values, computes per-k recall as `|top-k ∩ gold| / |gold|`.
4. It returns a flat `dict[str, float]` of macro-averaged recall@k.

## Integration Points

### Dependencies (inbound)

| Dependency | Type | Use |
|---|---|---|
| `gfmrag.evaluation.base_evaluator` | Internal | Abstract base class `BaseEvaluator` |
| Python stdlib `json` | External | JSONL deserialisation |
| Python stdlib `abc` | External | `ABC`, `abstractmethod` |
| Python stdlib `re`, `string` | External | Text normalisation |
| Python stdlib `collections.Counter` | External | Token overlap for F1 |
| Python stdlib `statistics.mean` | External | Macro-averaging in `RetrievalEvaluator` |

Notable: **zero PyPI dependencies** — no torch, numpy, or sklearn required.

### Consumers (outbound)

| Consumer module | Import path | Mechanism |
|---|---|---|
| `gfmrag.workflow.*` (QA / IRCoT scripts) | `from gfmrag.evaluation import HotpotQAEvaluator` | Direct instantiation + `evaluate()` call |
| Any external training/eval pipeline | `from gfmrag.evaluation import <EvaluatorClass>` | Same pattern |
| `gfmrag.__init__` (via `__all__`) | Re-exports via `gfmrag.evaluation` | Re-export chain |

### Exported API (`__init__.py`)

```python
from .base_evaluator import BaseEvaluator
from .hotpot_qa_evaluator import HotpotQAEvaluator
from .musique_evaluator import MusiqueEvaluator
from .retrieval_evaluator import RetrievalEvaluator
from .two_wiki_qa_evaluator import TwoWikiQAEvaluator
```

All five classes are re-exported via `__all__` and available as `gfmrag.evaluation.<ClassName>`.

### Architectural notes / technical debt

1. **Duplicated `normalize_answer`**: The identical ~12-line normalisation pipeline is copy-pasted across three QA evaluator files (`hotpot_qa_evaluator.py`, `musique_evaluator.py`, `two_wiki_qa_evaluator.py`). A common `_text_utils` module would eliminate the duplication.
2. **Different `exact_match_score` signatures**: `HotpotQAEvaluator` returns `int` (1 or 0), while `TwoWikiQAEvaluator` returns `bool`. Callers treat them interchangeably via `float(em)`, but the type inconsistency is a latent bug surface.
3. **`update_answer` side effects**: The function mutates the `metrics` dict passed by reference. This is an implicit dependency between the per-example loop and the accumulator.
4. **No serialisation**: The package has no facility to write results to disk — consumers must handle output serialisation themselves.
5. **Adapted code**: Three of the four evaluators are explicitly adapted from the HippoRAG repository (see file-level comments).
