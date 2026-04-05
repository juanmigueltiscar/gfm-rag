#!/bin/bash
# run_stage3_qa_ircot_inference.sh — Evaluar el sistema con preguntas de test (Stage 3)
#
# Ejecuta el sistema completo GFM-RAG sobre test.json y calcula métricas.
# Usa el modelo indicado en stage3_qa_ircot_inference.yaml (graph_retriever.model_path).
# Guarda resultados en: outputs/qa_agent_inference/
#
# Uso: ejecutar desde gfm-rag-main/
#   cd /home/compartit/gaiatec/GFM-new/gfm-rag-main
#   bash run_stage3_qa_ircot_inference.sh
#
# Para consultas interactivas (no evaluación), usa en su lugar:
#   uv run python ../gaiatec/scripts/examen.py --preguntas preguntas.txt
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

echo "Iniciando evaluación del sistema GFM-RAG..."
echo "Inicio: $(date '+%Y-%m-%d %H:%M:%S')"

uv run python -m gfmrag.workflow.stage3_qa_ircot_inference

echo "Completado: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Resultados en: outputs/qa_agent_inference/"
