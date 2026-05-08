#!/bin/bash
# Stage 2: Evaluate fine-tuned G-reasoner on master_ceramica dataset (single GPU).
# Run from the repository root: bash scripts/g-reasoner-ceramics/stage2_evaluate.sh
#
# CHECKPOINT must point to the 'pretrained/' directory produced by stage2_finetune.sh,
# e.g.:  outputs/qa_finetune/2025-05-07/12-00-00/pretrained
# That directory contains config.json (with input_dim=2048) and model.pth.
# The model architecture is fully reconstructed from config.json — no need to set
# model dims manually here.

# Load environment variables
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ─── Embedding dimension — must match what was used in stage1/stage2 ─────────
EMBED_DIM=2048

# ─── Checkpoint ──────────────────────────────────────────────────────────────
CHECKPOINT="${1:-}"          # Pass as first argument, or set directly below
# CHECKPOINT="outputs/qa_finetune/YYYY-MM-DD/HH-MM-SS/pretrained"

if [ -z "${CHECKPOINT}" ]; then
    echo "Usage: $0 <path_to_pretrained_dir>"
    echo "  e.g. $0 outputs/qa_finetune/2025-05-07/12-00-00/pretrained"
    exit 1
fi

# ─── Dataset ────────────────────────────────────────────────────────────────
DATA_ROOT="data"
EVAL_DATA_NAME="master_ceramica"

echo "Evaluating checkpoint: ${CHECKPOINT}"
echo "Dataset:               ${EVAL_DATA_NAME}"

HYDRA_FULL_ERROR=1 python -m gfmrag.workflow.sft_training \
    --config-path config/gfm_reasoner \
    --config-name sft_training \
    load_model_from_pretrained="${CHECKPOINT}" \
    datasets.cfgs.root="${DATA_ROOT}" \
    datasets.train_names=[] \
    datasets.valid_names=["${EVAL_DATA_NAME}"] \
    datasets.feat_dim="${EMBED_DIM}" \
    datasets.init_datasets=false \
    +datasets.cfgs.skip_empty_target=true \
    text_emb_model.text_emb_model_name=GaiatecEmbedder \
    text_emb_model.truncate_dim="${EMBED_DIM}" \
    text_emb_model.api_base=http://localhost:8083/v1 \
    trainer.args.do_train=false \
    trainer.args.do_eval=true \
    trainer.args.do_predict=true \
    trainer.args.eval_batch_size=1 \
    trainer.metrics=[hits@2,hits@5,recall@2,recall@5,mrr]
