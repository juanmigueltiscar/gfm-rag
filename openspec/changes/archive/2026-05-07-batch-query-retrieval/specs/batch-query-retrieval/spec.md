## ADDED Requirements

### Requirement: retrieve acepta lista de queries

`GFMRetriever.retrieve()` SHALL aceptar `query: str | list[str]`. Si es `str`, el comportamiento SHALL ser idéntico al actual. Si es `list[str]`, SHALL ejecutar batching y retornar una lista de resultados.

#### Scenario: Single query no cambia comportamiento
- **WHEN** se llama a `retriever.retrieve("test query", top_k=5)`
- **THEN** se retorna un `dict[str, list[dict]]` con las mismas claves y estructura que antes del cambio

#### Scenario: Batch de 3 queries retorna 3 resultados
- **WHEN** se llama a `retriever.retrieve(["Q1", "Q2", "Q3"], top_k=5)`
- **THEN** se retorna una `list` de 3 elementos, cada uno un `dict[str, list[dict]]` con las mismas claves y estructura que el caso single

#### Scenario: Batch de queries pasa target_types a cada query individual
- **WHEN** se llama a `retriever.retrieve(["Q1", "Q2"], top_k=3, target_types=["document", "entity"])`
- **THEN** cada dict de resultado contiene las claves `"document"` y `"entity"`

### Requirement: NER se ejecuta en paralelo para batch queries

Cuando `query` es `list[str]`, el NER SHALL ejecutarse en paralelo sobre las queries usando `ThreadPoolExecutor`, sin modificar la interfaz de `BaseNERModel`.

#### Scenario: NER llamado una vez por query en threads independientes
- **WHEN** se llama a `retrieve(["Q1", "Q2"], top_k=5)`
- **THEN** `self.ner_model.__call__` es invocado 2 veces, potencialmente de forma concurrente

#### Scenario: NER paralelo preserva el orden de resultados
- **WHEN** se llama a `retrieve(["Q1", "Q2", "Q3"], top_k=5)`
- **THEN** los resultados en la posición `i` corresponden a `queries[i]`, independientemente del orden de finalización de los threads

### Requirement: Entity Linking se ejecuta en una sola llamada para todas las entidades NER

Las entidades NER de todas las queries SHALL deduplicarse antes de llamar al EL, y las máscaras por query SHALL reconstruirse a partir del dict de resultados.

#### Scenario: Entidades compartidas entre queries se deduplican
- **WHEN** Q1 produce `["France"]` y Q2 produce `["France"]`
- **THEN** se llama a `el_model(["France"], topk=1)` una sola vez
- **AND** ambas queries usan el mismo resultado de EL para "France" en sus máscaras respectivas

#### Scenario: Entidades distintas entre queries se lotean juntas
- **WHEN** Q1 produce `["France"]` y Q2 produce `["Germany"]`
- **THEN** se llama a `el_model(["France", "Germany"], topk=1)` una sola vez
- **AND** la máscara de Q1 contiene solo `node2id("France")` y la de Q2 solo `node2id("Germany")`

### Requirement: Queries sin entidades generan máscara de ceros

Si `ner_model()` devuelve `[]` para una query, SHALL generarse `torch.zeros(num_nodes)` como máscara directamente, sin llamar al EL para esa query.

#### Scenario: Query sin entidades produce máscara de ceros
- **WHEN** `ner_model("some query")` retorna `[]`
- **THEN** la máscara para esa query es `torch.zeros(num_nodes)`
- **AND** no se llama a `el_model` con esa query como argumento

#### Scenario: Batch con algunas queries sin entidades funciona correctamente
- **WHEN** Q1 produce `["France"]`, Q2 produce `[]`, Q3 produce `["Germany"]`
- **THEN** `el_model` recibe `["France", "Germany"]` (sin incluir Q2)
- **AND** la máscara de Q2 es `torch.zeros(num_nodes)`
- **AND** las tres queries producen resultados válidos

### Requirement: max_batch_size se configura en __init__ y puede sobrescribirse en retrieve

`GFMRetriever.__init__` SHALL aceptar `max_batch_size: int = 4` y almacenarlo como `self.max_batch_size`. `retrieve()` SHALL aceptar `max_batch_size: int | None = None` y usar el valor de instancia cuando no se provea.

#### Scenario: max_batch_size desde __init__ se usa en retrieve
- **WHEN** `GFMRetriever(..., max_batch_size=8)` y se llama a `retrieve(["Q1", ..., "Q10"], top_k=5)`
- **THEN** el chunking usa `max_batch_size=8`, procesando 2 chunks (8 + 2)

#### Scenario: max_batch_size en retrieve sobrescribe el de __init__
- **WHEN** `GFMRetriever(..., max_batch_size=8)` y se llama a `retrieve(["Q1", ..., "Q10"], top_k=5, max_batch_size=3)`
- **THEN** el chunking usa `max_batch_size=3`, no el valor de instancia

#### Scenario: max_batch_size por defecto es 4
- **WHEN** se instancia `GFMRetriever(...)` sin `max_batch_size` y se llama a `retrieve([...], top_k=5)` sin `max_batch_size`
- **THEN** se usa `max_batch_size=4`

### Requirement: max_batch_size controla el chunking automático

El `max_batch_size` SHALL dividir la lista de queries en chunks que no excedan ese tamaño antes de ejecutar el GNN forward.

#### Scenario: Batch menor que max_batch_size se procesa en un solo chunk
- **WHEN** se llama a `retrieve(["Q1", "Q2"], top_k=5, max_batch_size=4)`
- **THEN** el GNN forward se ejecuta una sola vez con `batch_size=2`

#### Scenario: Batch mayor que max_batch_size se divide en chunks
- **WHEN** se llama a `retrieve(["Q1", "Q2", "Q3", "Q4", "Q5"], top_k=5, max_batch_size=2)`
- **THEN** el GNN forward se ejecuta 3 veces: `bs=2`, `bs=2`, `bs=1`
- **AND** los resultados se concatenan en orden preservando el mapping 1:1 con las queries de entrada

### Requirement: Chunking emite logs a nivel DEBUG

El bucle de chunking SHALL emitir `logger.debug()` indicando el chunk actual y el total de chunks cada vez que se procesa un chunk.

#### Scenario: Chunk único emite un mensaje de debug
- **WHEN** se procesa un batch de 3 queries con `max_batch_size=4`
- **THEN** se emite un mensaje `logger.debug` indicando "Processing chunk 1/1 (3 queries)"

#### Scenario: Múltiples chunks emiten mensajes secuenciales
- **WHEN** se procesa un batch de 5 queries con `max_batch_size=2`
- **THEN** se emiten mensajes `logger.debug` para "Processing chunk 1/3 (2 queries)", "chunk 2/3 (2 queries)", "chunk 3/3 (1 queries)"

### Requirement: Post-procesado usa indexado por query en vez de squeeze

El bucle de post-procesado SHALL iterar sobre `range(batch_size)` e indexar `pred[q_idx, node_ids]` en lugar de usar `squeeze(0)`.

#### Scenario: Cada query del batch obtiene sus propios top-k resultados
- **WHEN** `pred` tiene shape `(3, num_nodes)` con valores distintos por fila
- **THEN** `results[0]` refleja `pred[0]`, `results[1]` refleja `pred[1]`, `results[2]` refleja `pred[2]`

#### Scenario: top_k respetado por query individual
- **WHEN** un target type tiene 10 nodos y se pide `top_k=3`
- **THEN** cada query en el batch recibe exactamente 3 resultados para ese target type

### Requirement: El método existente prepare_input_for_graph_retriever se preserva sin cambios

El método `prepare_input_for_graph_retriever(self, query: str)` SHALL permanecer con la misma firma y comportamiento. La ruta single-query de `retrieve()` SHALL seguir usándolo.

#### Scenario: prepare_input_for_graph_retriever sigue funcionando igual
- **WHEN** se llama a `retriever.prepare_input_for_graph_retriever("test query")`
- **THEN** retorna un dict con claves `"question_embeddings"` y `"start_nodes_mask"` de shapes `(1, emb_dim)` y `(1, num_nodes)`
