## Why

Actualmente `GFMRetriever.retrieve()` procesa una query a la vez, pero el modelo GNN (`GNNRetriever`, `GraphReasoner`) ya soporta `batch_size > 1` de forma nativa — toda la pipeline de forward deriva el batch del input. El cuello de botella está en el preprocesado (`prepare_input_for_graph_retriever`), diseñado para un solo string. Pasar 3-4 queries en batch permitiría ejecutar el GNN, el text embedding y el entity linking en una sola pasada por GPU, reduciendo latencia total y overhead de ida/vuelta CPU-GPU.

## What Changes

- `GFMRetriever.retrieve()` acepta `str | list[str]` como primer argumento. Si es `str`, comportamiento actual sin cambios. Si es `list[str]`, ejecuta batching automático.
- `GFMRetriever.__init__()` y `from_index()` aceptan `max_batch_size` (default 4) para configurar el chunk size a nivel de instancia. `retrieve()` permite sobrescribirlo por llamada.
- `GFMRetriever.prepare_input_for_graph_retriever()` se complementa con un nuevo método `prepare_batch_input()` que construye `(bs, emb_dim)` y `(bs, num_nodes)` para el modelo.
- NER se ejecuta en paralelo vía `ThreadPoolExecutor` cuando hay múltiples queries, sin cambios en la interfaz `BaseNERModel`.
- Las queries sin entidades detectadas (NER vacío) generan máscara de ceros directamente, evitando pasar por entity linking.
- El chunking emite `logger.debug()` indicando progreso (chunk actual / total, número de queries).
- El post-procesamiento de resultados usa indexado por query (`pred[q_idx]`) en lugar de `squeeze(0)`.

## Capabilities

### New Capabilities
- `batch-query-retrieval`: Procesar múltiples queries en una sola pasada del modelo GNN, con NER paralelo, EL y text-embedding batcheados, y control de memoria GPU vía `max_batch_size`.

### Modified Capabilities
<!-- None - this is a purely additive change to the existing retrieval pipeline -->

## Impact

- **`gfmrag/gfmrag_retriever.py`**: `GFMRetriever` gana `prepare_batch_input()`, modifica `retrieve()` para aceptar `list[str]`, añade lógica de chunking.
- **`gfmrag/text_emb_models/base_model.py`**: sin cambios (ya acepta `list[str]`).
- **`gfmrag/graph_index_construction/ner_model/base_model.py`**: sin cambios en ABC (la parallelización es interna al retriever).
- **`gfmrag/graph_index_construction/entity_linking_model/`**: sin cambios (EL ya acepta listas de entidades).
- **`gfmrag/models/gfm_rag_v1/model.py`**: sin cambios (ya soporta `batch_size > 1`).
- **`gfmrag/models/gfm_reasoner/model.py`**: sin cambios (ya soporta `batch_size > 1`).
- **`tests/test_gfmrag_retriever.py`**: nuevos tests para `retrieve(queries_list)`.
- **No es breaking**: `retrieve(query: str)` sigue funcionando igual.

## Non-goals

- No se modifica la interfaz de `BaseNERModel` ni `BaseELModel`.
- No se implementa estimación dinámica de memoria GPU — se usa `max_batch_size` explícito.
- No se cambia el pipeline de entrenamiento (KGC/SFT).
- No se modifica `IRCoT` inference (sigue query a query, el batching es para retrieval directo).
