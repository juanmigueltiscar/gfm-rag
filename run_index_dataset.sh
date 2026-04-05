#!/bin/bash
# run_index_dataset.sh — Construir el Grafo de Conocimiento (Stage 1)
#
# Lee los documentos de bbdd/data_name/raw/dataset_corpus.json
# y extrae entidades y relaciones con NER (Mistral local) + OpenIE (Gemini).
# Genera: bbdd/data_name/processed/stage1/kg.txt
#
# Uso: ejecutar desde gfm-rag-main/
#   cd /home/compartit/gaiatec/GFM-new/gfm-rag-main
#   bash run_index_dataset.sh
#
# Ver pipeline.sh para una versión con menú interactivo y explicaciones.

set -euo pipefail

DATA_NAME="${1:-data_name}"  # Permite pasar el nombre del dataset como argumento

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

echo "Construyendo KG para dataset: $DATA_NAME"
echo "Inicio: $(date '+%Y-%m-%d %H:%M:%S')"

uv run python -m gfmrag.workflow.stage1_index_dataset \
    dataset.data_name="$DATA_NAME"

echo "Completado: $(date '+%Y-%m-%d %H:%M:%S')"
