# Scripts

## `test_retrieval.py`

Smoke-test end-to-end retrieval against a local vLLM stack using a tiny toy dataset (3 documents about France). Useful to verify that NER, OpenIE, entity linking, and the GFM-Retriever are all wired up correctly before indexing a real dataset.

**What it does:**

1. Creates `data/toy_raw/raw/documents.json` and `test.json` if they don't exist.
2. Checks that both vLLM servers (LLM + embeddings) are reachable.
3. Builds the knowledge graph for the toy dataset (NER → OpenIE → entity linking).
4. Instantiates `GFMRetriever` and runs both single-query and batch retrieval.
5. Prints ranked document results to stdout.

**Prerequisites:**

```
VLLM_BASE_URL=http://localhost:8082/v1    # LLM server (NER / OpenIE)
VLLM_MODEL=<model-name>
VLLM_EMBED_BASE_URL=http://localhost:8083/v1  # Embedding server
VLLM_EMBED_MODEL=<embed-model-name>
HF_TOKEN=<huggingface-token>             # Required to download G-reasoner weights
```

Set these in `.env` at the repository root or export them before running.

**Usage:**

```bash
.venv/bin/python scripts/test_retrieval.py
```

Expected output (abridged):

```
INFO | __main__ | vLLM LLM server is healthy
INFO | __main__ | vLLM embed server is healthy
INFO | __main__ | Building GFMRetriever from index (this may take a while on first run)...
============================================================
Single query: Who is the president of France?
============================================================
  #1  Emmanuel Macron            score=0.9231
  #2  France                     score=0.7814
  #3  Paris                      score=0.4102
============================================================
```

---

## `migrate_old_to_new.py`

Converts the old GFM-RAG stage-1 format (`kg.txt` + `document2entities.json`) into the current graph CSV format (`nodes.csv`, `edges.csv`, `relations.csv`).

**What it does:**

1. Reads `<old-dir>/processed/stage1/kg.txt` (lines: `subject, relation, object`) and normalizes entity names (diacritics stripped, lowercase).
2. Reads `<old-dir>/processed/stage1/document2entities.json` and generates `is_mentioned_in` edges.
3. Reads document content from `<old-dir>/raw/dataset_corpus.json`.
4. Parses Markdown `##`/`###` headings from document content and generates `has section` / `contains section` edges reflecting the heading hierarchy.
5. Optionally generates `equivalent` (synonymy) edges via cosine similarity using a vLLM embedding server.
6. Writes `nodes.csv`, `edges.csv`, `relations.csv` to `<out-dir>/processed/stage1/`.
7. Copies `<old-dir>/raw/` to `<out-dir>/raw/`, renaming `dataset_corpus.json` → `documents.json`.

**Prerequisites:**

Source data (under `--old-dir`):
- `processed/stage1/kg.txt`
- `processed/stage1/document2entities.json`
- `raw/dataset_corpus.json`

For synonymy edges (optional), set in `.env`:
```
VLLM_EMBED_BASE_URL=http://localhost:8083/v1
VLLM_EMBED_MODEL=<embed-model-name>
VLLM_API_KEY=EMPTY  # optional
```

**Usage:**

Quick run (no synonymy, useful for testing):
```bash
.venv/bin/python scripts/migrate_old_to_new.py --no-synonymy --force
```

Full migration with synonymy edges:
```bash
.venv/bin/python scripts/migrate_old_to_new.py --force
```

Custom paths:
```bash
.venv/bin/python scripts/migrate_old_to_new.py \
  --old-dir  data/master_ceramica_old_format \
  --out-dir  data/master_ceramica \
  --threshold 0.8 \
  --max-sim-neighbors 100 \
  --force
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--old-dir` | `data/master_ceramica_old_format` | Base directory of the old-format dataset |
| `--out-dir` | `data/master_ceramica` | Base directory of the new-format dataset |
| `--threshold` | `0.8` | Minimum cosine similarity for `equivalent` edges |
| `--max-sim-neighbors` | `100` | Maximum `equivalent` neighbours per entity |
| `--no-synonymy` | — | Skip `equivalent` edge generation (no vLLM needed) |
| `--force` | — | Overwrite existing output files |

**Expected output (--no-synonymy):**

```
INFO Loading inputs …
INFO   dataset_corpus.json: 393 entries
INFO   document2entities.json: 393 docs
INFO   kg.txt: 15051 triples loaded, 0 skipped
INFO Parsing markdown sections …
INFO   Section edges: 1291
INFO Total unique entity names: 12916
INFO Building nodes …
INFO Building edges …
INFO   Edges before synonymy: 32868
INFO Skipping synonymy edges (--no-synonymy)
INFO Writing CSVs to data/master_ceramica/processed/stage1/ …
INFO   nodes.csv: 13309 rows
INFO   edges.csv: 32868 rows
INFO   relations.csv: 3959 rows
INFO   raw: dataset_corpus.json -> documents.json
INFO   raw: test.json -> test.json
INFO   raw: train.json -> train.json
INFO Migration complete.
```
