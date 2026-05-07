## Context

`GFMRetriever.retrieve()` sigue un flujo secuencial para una sola query: NER → EL → text embedding → GNN forward → top-k post-procesado. La inspección del modelo revela que tanto `GNNRetriever.forward()` como `GraphReasoner.forward()` derivan `batch_size` de `question_embedding.size(0)` y operan correctamente para `bs > 1`. Las capas de rankeo (`SimpleRanker`, `IDFWeightedRanker`, etc.) también esperan shape `(batch_size, n_entities)`. El único bloqueo está en el preprocesado, donde `prepare_input_for_graph_retriever()` construye `(1, emb_dim)` y `(1, num_nodes)`.

## Goals / Non-Goals

**Goals:**
- `retrieve()` acepta `str | list[str]`, sin romper la API actual.
- NER paralelo sin tocar `BaseNERModel`.
- EL batcheado: todas las entidades NER (deduplicadas) en una sola llamada.
- Text embedding batcheado: `encode(queries)` de una vez.
- GNN forward con `bs > 1`, aprovechando el soporte existente en el modelo.
- Control de memoria GPU via `max_batch_size` con chunking interno.

**Non-Goals:**
- No modificar ABCs (`BaseNERModel`, `BaseELModel`, `BaseTextEmbModel`, `BaseGNNModel`).
- No estimación dinámica de memoria GPU.
- No batching en IRCoT (sigue query a query, el batching es para retrieval directo).
- No cambios en entrenamiento ni evaluación.

## Decisions

### 1. API: `retrieve(query: str | list[str])` en vez de `retrieve_batch(queries)`

Un solo método con type-based dispatch. Si `isinstance(query, str)`, se comporta exactamente como ahora (pasa por `prepare_input_for_graph_retriever`). Si es `list`, se usa la nueva ruta batcheada.

**Alternativa descartada**: método separado `retrieve_batch()`. Duplica lógica de post-procesado (top-k por target_type, construcción de dicts de resultado) y fuerza al caller a elegir entre dos métodos.

```
def retrieve(
    self,
    query: str | list[str],
    top_k: int,
    target_types: list[str] | None = None,
    max_batch_size: int = 4,
) -> dict[str, list[dict]] | list[dict[str, list[dict]]]:
```

Retorno:
- `str` → `dict[str, list[dict]]` (igual que ahora)
- `list[str]` → `list[dict[str, list[dict]]]` (uno por query de entrada)

### 2. NER paralelo interno con `ThreadPoolExecutor`

```python
def _ner_batch(self, queries: list[str]) -> list[list[str]]:
    results: list[list[str]] = [None] * len(queries)  # type: ignore[assignment]
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = {pool.submit(self.ner_model, q): i for i, q in enumerate(queries)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results
```

Cada query dispara su llamada LLM en un thread independiente. El servidor (OpenAI, vLLM) maneja la concurrencia. Sin cambios en `BaseNERModel.__call__(text: str) -> list`.

**Alternativa descartada**: `asyncio` + `aiohttp`. Añadiría una dependencia asíncrona que no existe en el proyecto y requeriría modificar el ABC de NER.

### 3. EL batcheado con deduplicación de entidades NER

```
Queries: Q1=["France", "Macron"], Q2=["France"], Q3=[]
  │
  ├─ unique_mentions = {"France", "Macron"}
  ├─ linked = el_model(list(unique_mentions), topk=1)
  │     → {"France": [{"entity": "France", ...}], "Macron": [...]}
  │
  ├─ mask_q1 = entities_to_mask([node2id("France"), node2id("Macron")])
  ├─ mask_q2 = entities_to_mask([node2id("France")])
  └─ mask_q3 = zeros(num_nodes)   ← NER vacío → máscara de ceros directa
```

El dict del EL es keyed por mención NER, así que entidades compartidas entre queries no causan conflicto.

### 4. Máscara de ceros para queries sin entidades

Si `ner_model()` devuelve `[]`, se genera `mask = torch.zeros(num_nodes)`. Esto evita pasar la query entera como "entidad" por EL (comportamiento actual de fallback, semánticamente frágil).

El modelo maneja boundary condition cero de forma natural: en `GNNRetriever.get_input_node_feature()`, todos los nodos empiezan en cero, y solo las entidades detectadas reciben el query embedding. Con máscara vacía, el input es `(bs, num_nodes, hidden)` todo ceros, pero `question_embedding` sigue fluyendo al modelo como contexto.

### 5. `max_batch_size` configurable en `__init__` y `from_index`, con override en `retrieve()`

`max_batch_size` se almacena como atributo de instancia en `__init__` (default 4). `from_index()` lo acepta y lo propaga. `retrieve()` acepta un override opcional. Si el caller no pasa el parámetro en `retrieve()`, se usa el valor de la instancia.

```
class GFMRetriever:
    def __init__(self, ..., max_batch_size: int = 4):
        ...
        self.max_batch_size = max_batch_size

    @staticmethod
    def from_index(..., max_batch_size: int = 4) -> "GFMRetriever":
        ...
        return GFMRetriever(..., max_batch_size=max_batch_size)

    def retrieve(self, query, top_k, target_types=None, max_batch_size=None):
        if max_batch_size is None:
            max_batch_size = self.max_batch_size
        if isinstance(query, str):
            return self._retrieve_single(query, top_k, target_types)

        all_results = []
        num_chunks = (len(query) + max_batch_size - 1) // max_batch_size
        for i in range(0, len(query), max_batch_size):
            chunk = query[i:i + max_batch_size]
            chunk_idx = i // max_batch_size + 1
            logger.debug("Processing chunk %d/%d (%d queries)", chunk_idx, num_chunks, len(chunk))
            chunk_results = self._retrieve_chunk(chunk, top_k, target_types)
            all_results.extend(chunk_results)
        return all_results
```

Esto permite que `from_index()` configure el `max_batch_size` adecuado para el hardware esperado (ej: 2 para GPU de 12GB, 8 para GPU de 48GB) sin que cada `retrieve()` tenga que repetirlo.

El chunking es transparente para el caller. Un `max_batch_size=4` para un grafo de ~100K nodos con hidden=512 consume ~800MB en float32 para intermedios, seguro en GPUs de 24GB+. Para grafos más grandes o GPUs más pequeñas, el caller ajusta.

**Alternativa descartada**: estimación dinámica con `torch.cuda.mem_get_info()`. Frágil: depende del número de capas del modelo, overhead de autograd, y fragmentación de memoria.

### 6. Post-procesado con indexado por query

```python
pred = self.graph_retriever(self.graph, graph_retriever_input)  # (bs, num_nodes)

results: list[dict] = []
for q_idx in range(pred.size(0)):
    query_results: dict[str, list[dict]] = {}
    for target_type in target_types:
        node_ids = self.graph.nodes_by_type[target_type]
        type_pred = pred[q_idx, node_ids]  # indexado, no squeeze
        topk = torch.topk(type_pred, k=min(top_k, len(node_ids)))
        query_results[target_type] = [...]
    results.append(query_results)
return results
```

La ruta single-query existente no se toca: sigue usando `prepare_input_for_graph_retriever` → `pred.squeeze(0)` como antes.

## Risks / Trade-offs

**[Risk] Memoria GPU en grafos grandes**
→ Mitigación: `max_batch_size` mantiene el batch acotado. El caller puede bajarlo a 1 o 2 si el grafo es enorme.

**[Risk] NER paralelo satura el servidor LLM**
→ Mitigación: `max_workers` en el `ThreadPoolExecutor` está limitado a `len(queries)`. Para batches grandes el chunking ya parte en grupos de `max_batch_size`. Si un servidor tiene rate-limiting, el caller debe configurar `max_batch_size` apropiadamente.

**[Risk] Queries sin entidades pueden degradar retrieval**
→ Mitigación: El modelo ya opera con boundary condition cero. En el comportamiento actual, el fallback produce el mismo efecto (máscara vacía si la query no matchea nada en el KG). El cambio solo elimina la llamada EL superflua.

**[Trade-off] La API retorna tipos distintos según entrada**
`str` → `dict`, `list[str]` → `list[dict]`. Esto es idiomático en Python (ej: `json.loads` según el input), pero el caller debe manejar ambos. Se documenta claramente.

## Open Questions

<!-- All resolved during design phase -->
