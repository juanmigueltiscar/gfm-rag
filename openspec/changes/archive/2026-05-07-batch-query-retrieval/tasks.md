## 1. Core: prepare_batch_input

- [x] 1.1 Añadir `_ner_batch(queries: list[str]) -> list[list[str]]` que ejecuta `self.ner_model` en paralelo con `ThreadPoolExecutor`, preservando el orden de resultados mediante un dict `futures -> index`.
- [x] 1.2 Añadir `prepare_batch_input(queries: list[str]) -> dict` que: (a) llama a `_ner_batch`, (b) deduplica entidades NER, (c) llama a `el_model` una sola vez con la lista única, (d) construye `start_nodes_mask` de shape `(bs, num_nodes)` usando `entities_to_mask` por query, (e) genera máscara de ceros para queries sin entidades (sin pasar por EL), y (f) llama a `text_emb_model.encode(queries, is_query=True)` para obtener `question_embeddings` de shape `(bs, emb_dim)`.
- [x] 1.3 Verificar que `prepare_input_for_graph_retriever` no se modifica — la ruta single-query sigue intacta.

## 2. Core: __init__, retrieve() dispatch y _retrieve_chunk()

- [x] 2.1 Añadir `max_batch_size: int = 4` al `__init__` de `GFMRetriever` y almacenarlo como `self.max_batch_size`.
- [x] 2.2 Modificar `retrieve(query, top_k, target_types=None, max_batch_size=None)` añadiendo `isinstance(query, str)` dispatch: si es `str`, ejecuta el mismo código de siempre (sin cambios). Si es `list[str]`, usa `max_batch_size or self.max_batch_size`, itera en chunks y emite `logger.debug("Processing chunk %d/%d (%d queries)", ...)` en cada iteración.
- [x] 2.3 Extraer `_retrieve_chunk(queries, top_k, target_types, query_utils)` que: (a) llama a `prepare_batch_input`, (b) mueve a device con `query_utils.cuda`, (c) ejecuta `self.graph_retriever(self.graph, input)`, y (d) post-procesa iterando `range(pred.size(0))` con indexado `pred[q_idx, node_ids]` en lugar de `squeeze(0)`. Retorna `list[dict[str, list[dict]]]`.
- [x] 2.4 Asegurar que el return type es correcto: `str` → `dict[str, list[dict]]`, `list[str]` → `list[dict[str, list[dict]]]`.

## 3. Tests

- [x] 3.1 Añadir test `test_retrieve_batch_single_query_unchanged`: verifica que `retrieve("query", top_k=2)` sigue devolviendo `dict` con los mismos valores que antes (reenfoque: usar mocks ya existentes con `graph_retriever.return_value = torch.tensor([[0.9, 0.1, 0.8, 0.2]])`).
- [x] 3.2 Añadir test `test_retrieve_batch_two_queries`: mock de NER con `side_effect` por query, mock de EL con dict de entidades, mock de text_emb con return de shape `(2, 128)`, mock de graph_retriever con `(2, 4)`. Verifica que `retrieve(["Q1", "Q2"], top_k=2)` retorna lista de 2 dicts, cada uno con `"document"`.
- [x] 3.3 Añadir test `test_retrieve_batch_entity_dedup`: Q1 NER → `["France"]`, Q2 NER → `["France"]`. Verifica que `el_model` se llama una sola vez con `["France"]`.
- [x] 3.4 Añadir test `test_retrieve_batch_empty_ner`: Q1 NER → `[]`, Q2 NER → `["France"]`. Verifica que EL recibe solo `["France"]`, que Q1 produce máscara de ceros, y que ambas queries producen resultados válidos.
- [x] 3.5 Añadir test `test_retrieve_batch_chunking`: 5 queries con `max_batch_size=2`. Verifica que `graph_retriever` se llama 3 veces (bs=2, bs=2, bs=1) y los resultados se devuelven en orden.
- [x] 3.6 Añadir test `test_max_batch_size_from_init`: instancia `GFMRetriever(..., max_batch_size=3)`, llama a `retrieve([Q1..Q5], top_k=2)` sin `max_batch_size`. Verifica que se usan chunks de 3.
- [x] 3.7 Añadir test `test_max_batch_size_override`: instancia con `max_batch_size=8`, llama a `retrieve` con `max_batch_size=2`. Verifica que el override tiene prioridad.
- [x] 3.8 Ejecutar todos los tests existentes de `test_gfmrag_retriever.py` para confirmar que la ruta single-query no tiene regresiones.
