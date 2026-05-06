# scripts/

## Responsibility

**Orchestration & Execution Launch-Pad** — the outermost entry-point layer for the entire GFM-RAG pipeline. All shell scripts here serve as thin executable wrappers that invoke Python modules (`python -m gfmrag.workflow.*` or `torchrun`) with Hydra-based configuration overrides. No application logic lives in this directory; its sole purpose is to parameterize and sequence invocations of the core `gfmrag` package for specific datasets, model variants, and pipeline stages.

Three sub-directories correspond to the three supported execution workflows:
| Sub-directory | Model Family | Target |
|---|---|---|
| `gfm-rag/` | GFM-RAG-8M | Full 3-stage pipeline (index → pretrain → finetune → QA) |
| `g-reasoner/` | G-reasoner-34M | 3-stage pipeline (index → finetune → evaluate → QA) |
| `graphrag_benchmark_evaluation/` | Both | GraphRAG Benchmark evaluation (CS, Medical, Novel splits) |

---

## Design Patterns

### 1. Pipeline (3-Stage Sequential)
Every workflow is organized into three sequential stages, enforced at the file-naming level and encoded in Hydra config trees:
- **Stage 1** (`stage1_data_index.sh`) — Raw document → graph index (NER/OpenIE → nodes.csv, relations.csv, edges.csv)
- **Stage 2** (`stage2_pretrain.sh`, `stage2_finetune.sh`, `stage2_evaluate.sh`, `stage2_retrieval.sh`) — KG completion training or SFT + retrieval evaluation
- **Stage 3** (`stage3_qa_inference.sh`, `stage3_qa_ircot_inference.sh`) — LLM-based QA inference from retrieved documents

### 2. Command Pattern (Shell as Launcher)
Each `.sh` file is a **Command object**: it captures a single executable intent with all parameters as top-level shell variables, then delegates to `python -m <module>` or `torchrun -m <module>`. The shell script is the minimum necessary glue — Hydra overrides are passed as `key=value` pairs on the CLI rather than modifying YAML files.

### 3. Hydra Composition & Override (Configuration Strategy)
All scripts use Hydra's CLI override mechanism to select and parameterize configs at runtime:
- `--config-path <path>` — selects config tree (`gfm_rag/` vs `gfm_reasoner/`)
- `--config-name <name>` — selects base YAML (e.g. `sft_training`, `stage3_qa_ircot_inference`)
- `key=value` overrides — every shell variable maps to a Hydra config key path (e.g. `llm.model_name_or_path=${LLM}`)

This is a pure **Strategy pattern**: the config path/name selects an algorithm variant (GFM-RAG vs G-reasoner, standard QA vs IRCoT), while CLI overrides inject context-specific parameters.

### 4. Worker Pool (multiprocessing.dummy.Pool)
Both `graphrag_bench_qa.py` and `graphrag_benchmark_qa.py` use a **ThreadPool** (`from multiprocessing.dummy import Pool`) for parallel LLM inference across samples. The number of worker threads is configured via `cfg.test.n_threads`.

### 5. Map-Reduce (Task Grouping)
The benchmark QA modules implement a two-phase **Map-Reduce**:
- **Map**: `resolve_task_type()` classifies each sample into a task type (e.g., FB/MC/MS/OE/TF for GraphRAG-Bench CS; Fact Retrieval/Complex Reasoning/etc. for Medical/Novel)
- **Group**: samples are partitioned into `task_groups` (a `dict[str, list[dict]]`)
- **Reduce per group**: each group is processed with a separate `QAPromptBuilder` instance loaded from its task-specific YAML prompt config

### 6. Factory Method (LLM Instantiation)
LLM backends are created via `hydra.utils.instantiate(cfg.llm)`, which resolves `_target_: gfmrag.llms.ChatGPT` (or any registered class) from the Hydra config. This decouples shell scripts from LLM implementation details.

### 7. Strategy for Task Type Resolution
`prompt_utils.resolve_task_type()` supports two resolution strategies selected by `cfg.task_type.source`:
- `id_prefix`: parses `sample["id"]` using a delimiter (e.g., `FB-001` → `FB`)
- `field`: reads a named field directly from the sample dict (e.g., `sample["question_type"]`)

### 8. Batch / Sequential Dataset Iteration
Multiple scripts iterate over dataset shards using shell-loop enumeration:
```
START_N=0; END_N=19
for i in $(seq ${START_N} ${END_N}); do ... dataset.data_name=${DATA_NAME}${i}
```
This batches a large dataset (20 shards × 3 source datasets = 60 index entries) into sequential `python -m` invocations, avoiding OOM during index construction.

---

## Data & Control Flow

### Entry Points

All scripts ultimately call one of these Python module entry points:

| Module | Invocation | Stage | Purpose |
|---|---|---|---|
| `gfmrag.workflow.index_dataset` | `python -m` | 1 | Raw doc → graph index (NER, OpenIE, graph construction) |
| `gfmrag.workflow.kgc_training` | `torchrun` | 2 | Large-scale KG completion pretraining (GFM-RAG only) |
| `gfmrag.workflow.sft_training` | `torchrun` | 2 | Supervised fine-tuning + retrieval eval (both models) |
| `gfmrag.workflow.qa` | `python -m` | 3 | Standard QA: retrieved docs → LLM → answer |
| `gfmrag.workflow.qa_ircot_inference` | `python -m` | 3 | IRCoT: iterative retrieval + reasoning loop |
| `graphrag_benchmark.graphrag_bench_qa` | `python -m` | 3* | GraphRAG-Bench CS evaluation |
| `graphrag_benchmark.graphrag_benchmark_qa` | `python -m` | 3* | GraphRAG-Bench Medical/Novel evaluation |

### GFM-RAG Pipeline (scripts/gfm-rag/)

```
stage1_data_index.sh
  └─ python -m gfmrag.workflow.index_dataset
       Input:  raw documents at data/{dataset}/raw/
       Output: stage1 CSVs (nodes.csv, relations.csv, edges.csv)
                stage2 PT files (graph.pt, node2id.json, rel2id.json)

stage2_pretrain.sh
  └─ torchrun -m gfmrag.workflow.kgc_training
       Input:  stage2 PT files (all 60 train shards concatenated)
       Output: trained KG completion embeddings

stage2_finetune.sh
  └─ torchrun -m gfmrag.workflow.sft_training
       Input:  pretrained checkpoint + stage2 PT files
       Output: fine-tuned GFM-RAG model + predictions_*.json

stage3_qa_inference.sh
  └─ python -m gfmrag.workflow.qa
       Input:  predictions JSON + nodes.csv
       Data Flow:
         1. Load LLM (gpt-4o-mini via hydra instantiate)
         2. Load retrieval results (documents at top_k)
         3. For each sample: build prompt → LLM.generate_sentence() → collect answer
       Output: QA predictions

stage3_qa_ircot_inference.sh
  └─ python -m gfmrag.workflow.qa_ircot_inference
       Input:  graph_retriever.model_path + LLM
       Data Flow:
         1. Graph retriever loaded from model_path
         2. For each question (max_steps iterations):
            a. Retrieve subgraph
            b. LLM reasons + generates next query
            c. Repeat until answer found or max_steps reached
       Output: IRCoT QA predictions
```

### G-reasoner Pipeline (scripts/g-reasoner/)

```
stage1_data_index.sh   (same as GFM-RAG but uses --config-path config/gfm_reasoner)
stage2_finetune.sh     (same entry point, different config tree + additional params)
stage2_evaluate.sh     (sft_training in eval-only mode: do_train=false, do_eval=true, do_predict=true)
stage3_qa_inference.sh (same workflow.qa as GFM-RAG)
stage3_qa_ircot_inference.sh (same workflow.qa_ircot_inference but uses own config)
```

Key differences from GFM-RAG:
- G-reasoner uses a single `sft_training` entry point (no separate `kgc_training` pretrain step)
- Config tree is `config/gfm_reasoner` instead of default (`gfm_rag`)
- Additional params: `split_graph_training`, `split_graph_inference`, `split_graph_partition` (metis/contiguous)
- Text embedding model configurable via `text_emb_model` shell env var
- Training data filter controlled by `sft_constructor.enable_filtering`

### GraphRAG Benchmark Evaluation (scripts/graphrag_benchmark_evaluation/)

```
stage1_data_index.sh
  └─ python -m gfmrag.workflow.index_dataset (3 benchmark datasets: graphrag_bench_cs, graphrag_benchmark_medical, graphrag_benchmark_novel)

stage2_retrieval.sh
  └─ torchrun -m gfmrag.workflow.sft_training (eval-only, both config trees)
       Runs retrieval for both GFM-RAG and G-reasoner checkpoints on all 3 benchmark datasets
       Output: predictions_*.json files

stage3_qa_inference_graphrag_bench.sh
  └─ python -m graphrag_benchmark.graphrag_bench_qa
       Input:  config=graphrag_bench_cs_qa_inference, retrieval results, documents.json
       Data Flow:
         1. Load retrieval results JSON
         2. Load documents.json (title -> content map)
         3. resolve_task_type() via id_prefix (FB/MC/MS/OE/TF)
         4. Task-group samples into prompt_builders per group
         5. ThreadPool parallel: for each sample:
            a. Map predicted doc titles → document content
            b. Build prompt via QAPromptBuilder.build_input_prompt()
            c. LLM.generate_sentence()
            d. Save per-task JSON output
       Output: Per-task JSON files (e.g. GraphRAG-Bench_FB.json)

stage3_qa_inference_graphrag_benchmark.sh
  └─ python -m graphrag_benchmark.graphrag_benchmark_qa (runs twice: medical + novel)
       Input:  config=graphrag_benchmark_qa_inference, retrieval results
       Data Flow: (same structure as graphrag_bench_qa but)
         - resolve_task_type() via field source (question_type field)
         - Output format is JSONL (one prediction per line)
         - Post-processes response for "Answer:" prefix to extract generated_answer
       Output: prediction.jsonl per dataset split
```

---

## Integration Points

### Hydra Config Links
| Path | Referenced By | Purpose |
|---|---|---|
| `pkg://gfmrag.workflow.config` | All benchmark YAML configs | Shared Hydra search path for prompt configs |
| `gfmrag/workflow/config/gfm_rag/` | `gfm-rag/*.sh` (implicit default) | GFM-RAG config tree |
| `gfmrag/workflow/config/gfm_reasoner/` | `g-reasoner/*.sh` (`--config-path config/gfm_reasoner`) | G-reasoner config tree |

### Core Package Entry Points
| Python Module | Invoked By | Mechanism |
|---|---|---|
| `gfmrag.workflow.index_dataset` | All `stage1_*.sh` files | `python -m gfmrag.workflow.index_dataset` |
| `gfmrag.workflow.kgc_training` | `gfm-rag/stage2_pretrain.sh` | `torchrun -m gfmrag.workflow.kgc_training` |
| `gfmrag.workflow.sft_training` | `gfm-rag/stage2_finetune.sh`, `g-reasoner/stage2_finetune.sh`, `g-reasoner/stage2_evaluate.sh`, `graphrag_benchmark/scripts/stage2_retrieval.sh` | `torchrun -m gfmrag.workflow.sft_training` |
| `gfmrag.workflow.qa` | `gfm-rag/stage3_qa_inference.sh`, `g-reasoner/stage3_qa_inference.sh` | `python -m gfmrag.workflow.qa` |
| `gfmrag.workflow.qa_ircot_inference` | `gfm-rag/stage3_qa_ircot_inference.sh`, `g-reasoner/stage3_qa_ircot_inference.sh` | `python -m gfmrag.workflow.qa_ircot_inference` |

### Library Dependencies (from Python modules)
- `gfmrag.llms.ChatGPT` (via `hydra.utils.instantiate(cfg.llm)`) — LLM inference backend
- `gfmrag.prompt_builder.QAPromptBuilder` — Prompt construction from YAML configs
- `gfmrag.prompt_builder.build_input_prompt(question, retrieved_result)` — Core prompt assembly API
- `gfmrag.utils.get_rank()` — Distributed seed computation
- `omegaconf.DictConfig, OmegaConf` — Config loading/serialization
- `hydra.utils.to_absolute_path` — Config path resolution
- `hydra.core.hydra_config.HydraConfig` — Runtime output directory discovery

### File System Integration Points
| Path Pattern | Producer | Consumer | Format |
|---|---|---|---|
| `data/{dataset}/raw/documents.json` | External (downloaded) | `stage1_data_index.sh`, QA scripts | JSON (doc id → content) |
| `data/{dataset}/processed/stage1/{nodes,relations,edges}.csv` | `index_dataset` | `sft_training`, `qa` scripts | CSV |
| `data/{dataset}/processed/stage2/{graph.pt,node2id.json,rel2id.json}` | `index_dataset` | `kgc_training`, `sft_training` | PyTorch + JSON |
| `outputs/qa_finetune/latest/predictions_{dataset}_test.json` | `sft_training` (eval mode) | `workflow.qa` scripts, benchmark QA scripts | JSON |
| `outputs/qa_inference/{date}/{time}/{task}/` | QA scripts | Human evaluation | JSON/JSONL |

### Environment Variable Dependencies
| Variable | Used In | Purpose |
|---|---|---|
| `HYDRA_FULL_ERROR=1` | All `.sh` scripts | Enable full Hydra error tracebacks |
| `CUDA_VISIBLE_DEVICES` | `graphrag_benchmark_evaluation/scripts/stage1_data_index.sh` | GPU selection for index building |
| `TEXT_EMBEDDING_MODEL` | `g-reasoner/stage1_data_index.sh` | Text embedding model for G-reasoner |
| `ENABLE_FILTERING` | `g-reasoner/stage1_data_index.sh` | SFT constructor filter toggle |
| `TRAIN_MODE` | `g-reasoner/stage2_finetune.sh` | Training mode string |
| `CHECKPOINT` / `PRETRAINED` | Finetune scripts | Resume from checkpoint or load pretrained weights |
| `PATH_TO_YOUR_CHECKPOINT` | `graphrag_benchmark_evaluation/scripts/stage2_retrieval.sh` | Placeholder for user checkpoint path |
| `OPENAI_API_KEY` | Runtime (via `gfmrag.llms.ChatGPT`) | LLM API authentication |
| `HF_TOKEN` | Runtime (via model loading) | HuggingFace gated model access |
