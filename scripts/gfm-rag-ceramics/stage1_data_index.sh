#!/bin/bash
# Stage 1: Build knowledge graph index for master_ceramica dataset (GFM-RAG pipeline).
# Run from the repository root: bash scripts/gfm-rag-ceramics/stage1_data_index.sh

# Load environment variables (.env must define VLLM_EMBED_BASE_URL, VLLM_EMBED_MODEL,
# VLLM_BASE_URL, VLLM_MODEL for NER/OpenIE, and optionally VLLM_API_KEY).
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ─── Embedding model configuration ──────────────────────────────────────────
EMBED_URL="${VLLM_EMBED_BASE_URL:-http://localhost:8083/v1}"
EMBED_MODEL="${VLLM_EMBED_MODEL}"

# ─── LLM for NER / OpenIE ────────────────────────────────────────────────────
LLM_URL="${VLLM_BASE_URL:-http://localhost:8082/v1}"
LLM_MODEL="${VLLM_MODEL}"

# ─── Dataset ────────────────────────────────────────────────────────────────
DATA_ROOT="data"
DATA_NAME="master_ceramica"

echo "Indexing dataset: ${DATA_NAME}"
echo "Embedding model:  ${EMBED_MODEL} @ ${EMBED_URL}"
echo "LLM (NER/OpenIE): ${LLM_MODEL} @ ${LLM_URL}"

python -m gfmrag.workflow.index_dataset \
    --config-path config/gfm_rag \
    dataset.root="${DATA_ROOT}" \
    dataset.data_name="${DATA_NAME}" \
    ner_model.llm_api=vllm \
    ner_model.model_name="${LLM_MODEL}" \
    ner_model.base_url="${LLM_URL}" \
    openie_model.llm_api=vllm \
    openie_model.model_name="${LLM_MODEL}" \
    openie_model.base_url="${LLM_URL}" \
    el_model=vllm_el_model \
    el_model.api_base="${EMBED_URL}" \
    el_model.model_name="${EMBED_MODEL}"
