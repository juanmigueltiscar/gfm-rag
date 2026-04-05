#!/bin/bash
# run_state2_qa_finetune.sh — Entrenar el modelo GNN (Stage 2)
#
# Afina la red neuronal de grafos (GNN) usando los pares QA de train.json.
# El modelo aprende a navegar el KG para recuperar documentos relevantes.
# Guarda checkpoints en: outputs/qa_finetune/{fecha}/
#
# Uso: ejecutar desde gfm-rag-main/
#   cd /home/compartit/gaiatec/GFM-new/gfm-rag-main
#   bash run_state2_qa_finetune.sh
#
# Tras entrenar, actualiza model_path en:
#   gfmrag/workflow/config/stage3_qa_ircot_inference.yaml
#
# Ver pipeline.sh para una versión con menú interactivo y explicaciones.

set -euo pipefail

# Configurar nvjitlink dinámicamente
NVJITLINK_PATH=$(python -c "
import importlib.util, os
spec = importlib.util.find_spec('nvidia.nvjitlink')
if spec and spec.origin:
    print(os.path.join(os.path.dirname(spec.origin), 'lib'))
" 2>/dev/null || true)

if [ -n "$NVJITLINK_PATH" ] && [ -d "$NVJITLINK_PATH" ]; then
    export LD_LIBRARY_PATH="${NVJITLINK_PATH}:${LD_LIBRARY_PATH:-}"
fi

echo "Iniciando entrenamiento del modelo GNN..."
echo "Inicio: $(date '+%Y-%m-%d %H:%M:%S')"

uv run python -m gfmrag.workflow.stage2_qa_finetune

echo "Completado: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Checkpoints guardados en: outputs/qa_finetune/"
echo "Actualiza model_path en gfmrag/workflow/config/stage3_qa_ircot_inference.yaml"
