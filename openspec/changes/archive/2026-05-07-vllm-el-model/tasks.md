## 1. Implementación de VLLMELModel

- [x] 1.1 Crear `gfmrag/graph_index_construction/entity_linking_model/vllm_el_model.py` con la clase `VLLMELModel(BaseELModel)`: constructor con parámetros `model_name`, `api_base`, `api_key`, `root`, `use_cache`, `normalize`, `batch_size`, `query_instruct`, `passage_instruct`, `vllm_timeout`; health check en `__init__`; método `_add_instruct(instruct, texts)`
- [x] 1.2 Implementar `VLLMELModel.index(entity_list)`: caché MD5 en disco, llamada a `/v1/embeddings` en batches, normalización L2 opcional, almacenamiento en `self.entity_embeddings`
- [x] 1.3 Implementar `VLLMELModel.__call__(ner_entity_list, topk)`: embeddings de queries con `query_instruct`, similitud coseno con `self.entity_embeddings`, retorno de dict con `entity`/`score`/`norm_score`

## 2. Registro y configuración

- [x] 2.1 Añadir `VLLMELModel` a `gfmrag/graph_index_construction/entity_linking_model/__init__.py` (import + `__all__`)
- [x] 2.2 Crear `gfmrag/workflow/config/el_model/vllm_el_model.yaml` con `_target_`, `model_name`, `api_base`, `api_key`, `root`, `use_cache`, `normalize`, `batch_size`, `query_instruct`, `passage_instruct`

## 3. Tests

- [x] 3.1 Crear `tests/test_vllm_el_model.py`: test de inicialización correcta (mock del health check), test de fallo cuando el servidor no responde, test de `index()` con mock del cliente OpenAI, test de `__call__()` con embeddings conocidos, test de carga desde caché
