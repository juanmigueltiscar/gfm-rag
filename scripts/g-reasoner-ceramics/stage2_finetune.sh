#!/bin/bash
# Stage 2: Fine-tune G-reasoner on master_ceramica dataset (single GPU).
# Run from the repository root: bash scripts/g-reasoner-ceramics/stage2_finetune.sh
#
# Pretrained weights are loaded with strict=False: compatible GNN layers
# initialize from rmanluo/G-reasoner-34M; the input projection layer (shape
# changes from 1024->1024 to 2048->1024) is randomly initialized and learned.

# Load environment variables
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# ─── Embedding model configuration ──────────────────────────────────────────
EMBED_DIM=2048               # Change this if you switch to a different embedding model
EMBED_URL="${VLLM_EMBED_BASE_URL:-http://localhost:8000/v1}"
EMBED_MODEL="${VLLM_EMBED_MODEL}"

# ─── GNN architecture ────────────────────────────────────────────────────────
N_LAYERS_DIM=1024            # Internal GNN hidden dims (invariant across embedding models)
N_LAYERS="[${N_LAYERS_DIM},${N_LAYERS_DIM},${N_LAYERS_DIM},${N_LAYERS_DIM},${N_LAYERS_DIM},${N_LAYERS_DIM}]"

# ─── Training hyperparameters ────────────────────────────────────────────────
N_EPOCH=10
BATCH_SIZE=4
CHECKPOINT=null
PRETRAINED_WEIGHTS="rmanluo/G-reasoner-34M"
SAVE_BEST_ONLY=true
SAVE_PRETRAINED=true
USE_WANDB=false
RUN_NAME="stage2-finetune-g-reasoner-ceramics"

# ─── Dataset ────────────────────────────────────────────────────────────────
DATA_ROOT="data"
TRAIN_DATA_NAME="master_ceramica"
VALID_DATA_NAME="master_ceramica"
INIT_DATASETS=false          # Use pre-computed embeddings from stage1
MAX_DATA_IN_MEMORY=2
DATA_LOADING_WORKER=2

SPLIT_GRAPH_TRAINING=false
SPLIT_GRAPH_INFERENCE=false
SPLIT_GRAPH_METHOD="metis"

echo "Fine-tuning G-reasoner on: ${TRAIN_DATA_NAME}"
echo "Embedding model:           ${EMBED_MODEL} @ ${EMBED_URL} (dim=${EMBED_DIM})"
echo "Pretrained weights:        ${PRETRAINED_WEIGHTS} (strict=False)"

HYDRA_FULL_ERROR=1 python -m gfmrag.workflow.sft_training \
    --config-path config/gfm_reasoner \
    save_pretrained="${SAVE_PRETRAINED}" \
    wandb.enabled="${USE_WANDB}" \
    wandb.name="${RUN_NAME}" \
    model.entity_model.input_dim="${N_LAYERS_DIM}" \
    model.entity_model.hidden_dims="${N_LAYERS}" \
    +load_pretrained_weights="${PRETRAINED_WEIGHTS}" \
    load_model_from_pretrained=null \
    trainer.args.resume_from_checkpoint="${CHECKPOINT}" \
    datasets.cfgs.root="${DATA_ROOT}" \
    datasets.train_names=["${TRAIN_DATA_NAME}"] \
    datasets.valid_names=["${VALID_DATA_NAME}"] \
    datasets.feat_dim="${EMBED_DIM}" \
    datasets.init_datasets="${INIT_DATASETS}" \
    datasets.max_datasets_in_memory="${MAX_DATA_IN_MEMORY}" \
    datasets.data_loading_workers="${DATA_LOADING_WORKER}" \
    text_emb_model.api_base="${EMBED_URL}" \
    text_emb_model.text_emb_model_name="${EMBED_MODEL}" \
    text_emb_model.truncate_dim="${EMBED_DIM}" \
    trainer.args.num_epoch="${N_EPOCH}" \
    trainer.args.save_best_only="${SAVE_BEST_ONLY}" \
    trainer.args.eval_batch_size="${BATCH_SIZE}" \
    trainer.args.train_batch_size="${BATCH_SIZE}" \
    trainer.args.split_graph_training="${SPLIT_GRAPH_TRAINING}" \
    trainer.args.split_graph_inference="${SPLIT_GRAPH_INFERENCE}" \
    trainer.args.split_graph_partition="${SPLIT_GRAPH_METHOD}"
