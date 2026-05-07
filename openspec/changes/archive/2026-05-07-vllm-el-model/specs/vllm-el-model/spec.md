## ADDED Requirements

### Requirement: VLLMELModel conecta a un servidor vLLM remoto
`VLLMELModel` SHALL implementar `BaseELModel` y comunicarse con un servidor vLLM externo via la API OpenAI-compatible (`/v1/embeddings`). El paquete `vllm` NO debe ser una dependencia requerida; solo se necesita `openai`.

#### Scenario: Inicialización con servidor disponible
- **WHEN** se instancia `VLLMELModel(model_name="nvidia/llama-nemotron-embed-1b-v2", api_base="http://localhost:8000/v1")`
- **THEN** el objeto se crea correctamente y `self.client` es un `OpenAI` client configurado con `base_url=api_base`

#### Scenario: Inicialización con servidor no disponible
- **WHEN** se instancia `VLLMELModel` con un `api_base` cuyo endpoint `/health` no responde con 200
- **THEN** se lanza `RuntimeError` con un mensaje indicando que el servidor vLLM no está disponible

### Requirement: VLLMELModel indexa entidades obteniendo embeddings del servidor remoto
El método `index(entity_list)` SHALL llamar a `/v1/embeddings` para obtener los vectores de las entidades y almacenarlos en `self.entity_embeddings` como `torch.Tensor`.

#### Scenario: Indexado sin caché previa
- **WHEN** se llama a `model.index(["Paris", "London", "Berlin"])` y no existe caché
- **THEN** se realizan llamadas al servidor vLLM en batches, `self.entity_embeddings` contiene un tensor de shape `(3, embedding_dim)` y, si `use_cache=True`, el tensor se guarda en disco

#### Scenario: Indexado con caché existente
- **WHEN** se llama a `model.index(entity_list)` y existe un fichero de caché con el MD5 de la lista
- **THEN** los embeddings se cargan desde disco sin llamar al servidor vLLM

### Requirement: VLLMELModel enlaza entidades NER con las entidades indexadas
El método `__call__(ner_entity_list, topk)` SHALL obtener embeddings de las entidades NER, calcular similitud coseno contra `self.entity_embeddings` y retornar el top-k.

#### Scenario: Entity linking básico
- **WHEN** se llama a `model(["paris city"], topk=2)` tras indexar `["Paris", "London", "Berlin"]`
- **THEN** retorna un dict con clave `"paris city"` mapeando a una lista de hasta 2 dicts `{"entity": ..., "score": ..., "norm_score": ...}`

#### Scenario: norm_score del top resultado es siempre 1.0
- **WHEN** se retornan resultados con `topk >= 1`
- **THEN** el `norm_score` del primer resultado (mayor score) es `1.0` y los demás son relativos a él

### Requirement: VLLMELModel soporta instrucciones de query y passage
`VLLMELModel` SHALL aceptar los parámetros `query_instruct` y `passage_instruct`. Cuando se proporcionan, se concatenan como prefijo a los textos antes de enviarlos al servidor.

#### Scenario: Prefijo de query aplicado en __call__
- **WHEN** `query_instruct="Query: "` y se llama a `model(["Paris"])`
- **THEN** el texto enviado al servidor es `"Query: Paris"`

#### Scenario: Prefijo de passage aplicado en index
- **WHEN** `passage_instruct="Passage: "` y se llama a `model.index(["Paris"])`
- **THEN** el texto enviado al servidor es `"Passage: Paris"`

#### Scenario: Sin instrucciones, texto enviado sin modificar
- **WHEN** `query_instruct=None` y `passage_instruct=None`
- **THEN** los textos se envían al servidor tal cual, sin prefijo

### Requirement: VLLMELModel normaliza embeddings opcionalmente
Cuando `normalize=True`, `VLLMELModel` SHALL aplicar normalización L2 a los embeddings recuperados del servidor antes de almacenarlos o usarlos en la similitud coseno.

#### Scenario: Normalización activada
- **WHEN** `normalize=True` y se obtienen embeddings del servidor
- **THEN** la norma L2 de cada vector de embedding es aproximadamente `1.0`

#### Scenario: Normalización desactivada
- **WHEN** `normalize=False`
- **THEN** los embeddings se usan tal cual los devuelve el servidor

### Requirement: Hydra config disponible para VLLMELModel
SHALL existir un fichero `gfmrag/workflow/config/el_model/vllm_el_model.yaml` que permita instanciar `VLLMELModel` via `hydra.utils.instantiate`.

#### Scenario: Instanciación via Hydra
- **WHEN** se carga `el_model=vllm_el_model` en un workflow Hydra
- **THEN** se crea correctamente una instancia de `VLLMELModel` con los parámetros del YAML
