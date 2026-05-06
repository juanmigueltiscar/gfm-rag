## Context

El proyecto `gfmrag` tiene dos capas de integración con LLMs:

1. **Capa `gfmrag/llms/`** — Strategy pattern con `BaseLanguageModel` (ABC). Usada por los workflows de QA e IRCoT vía Hydra `_target_`. Implementaciones actuales: `ChatGPT` (OpenAI SDK), `HfCausalModel` (transformers local), `Gemini`, `MistralSmall32API`.

2. **Capa `langchain_util.py`** — Factoría `init_langchain_model()` que devuelve clientes LangChain (`ChatOpenAI`, `ChatNVIDIA`, `ChatTogether`, `ChatOllama`, `ChatLlamaCpp`). Usada por `LLMNERModel` y `LLMOPENIEModel` en la construcción de grafos.

El servidor vLLM expone una API OpenAI-compatible en `{base_url}/v1`. El proyecto ya usa este patrón en la capa de embeddings: `Qwen3TextEmbModel` instancia `OpenAI(api_key=..., base_url=...)` y hace health check en `__init__`.

El usuario tiene un servidor vLLM corriendo en producción, sin guided decoding habilitado, y quiere usar sus modelos en todas las capas del proyecto.

## Goals / Non-Goals

**Goals:**
- Implementar `VLLMModel(BaseLanguageModel)` para QA/IRCoT, usando el SDK de OpenAI con `base_url` configurable
- Añadir soporte `"vllm"` en `init_langchain_model()` para NER/OpenIE vía `ChatOpenAI(base_url=...)`
- Desacoplar JSON mode del tipo concreto de cliente en NER/OpenIE mediante bandera `json_mode`
- Health check estricto al init (crash si servidor no disponible)
- Tokenización exacta vía endpoint `/tokenize` de vLLM
- Todos los parámetros de conexión vía YAML config, no env vars

**Non-Goals:**
- No modificar `ChatGPT`, `HfCausalModel` ni otras implementaciones existentes
- No arrancar servidor vLLM desde el proyecto (debe estar corriendo)
- No modificar la capa de embeddings
- No añadir nuevas dependencias

## Decisions

### D1: Clase nueva `VLLMModel` en vez de reutilizar `ChatGPT` con `base_url`

**Alternativas consideradas:**
- **Opción A**: Añadir `base_url` a `ChatGPT` — descartado porque `get_token_limit()` está hardcodeado para modelos `gpt-*`, `tiktoken` falla con modelos no-OpenAI, y la semántica de "ChatGPT" no aplica a un servidor vLLM.
- **Opción C**: Crear `OpenAICompatibleModel` como base compartida y refactorizar `ChatGPT` — descartado por scope excesivo; `ChatGPT` tiene acoplamiento profundo a OpenAI (tiktoken, get_token_limit) que requeriría reescribir.

**Decisión**: Nueva clase independiente `VLLMModel(BaseLanguageModel)`. Sigue el patrón de `HfCausalModel` (implementación limpia sin tocar `ChatGPT`). Se registra en `__init__.py` y se usa vía `_target_: gfmrag.llms.VLLMModel`.

### D2: Tokenización vía endpoint `/tokenize` en vez de `AutoTokenizer` local

**Alternativas consideradas:**
- `AutoTokenizer.from_pretrained()` — ya es dependencia del proyecto y `HfCausalModel` lo usa. Sin embargo, puede diferir de la tokenización del servidor vLLM (configuraciones de chat template, tokens especiales). Requiere `HF_TOKEN` para modelos gated y descarga el tokenizer en cada init.
- Endpoint `/tokenize` de vLLM — el servidor vLLM lo expone nativamente. Devuelve el conteo exacto usado por el motor de inferencia. Añade un round-trip HTTP por llamada, pero garantiza consistencia.

**Decisión**: `/tokenize` de vLLM. Para producción con servidor ya corriendo, la consistencia exacta pesa más que la latencia de red. La llamada es ligera (solo tokenización, no inferencia).

### D3: Bandera `json_mode` explícita en vez de `isinstance(client, ChatOpenAI)` o atributo centinela

**Alternativas consideradas:**
- **Atributo centinela** (`client._supports_json_mode = False`): mínima invasión pero frágil y "mágico".
- **Cambiar firma de `init_langchain_model`** para devolver tupla `(client, json_mode)`: rompe la API existente.
- **Bandera explícita `json_mode` en constructores**: más cambios pero diseño limpio y desacoplado.

**Decisión**: Bandera `json_mode: bool = True` en `LLMNERModel.__init__` y `LLMOPENIEModel.__init__`. Se añade a los Hydra configs YAML con default `true` (preserva comportamiento actual). Los checks `isinstance(self.client, ChatOpenAI)` se reemplazan por `self.json_mode`. Esto desacopla una capacidad (JSON mode) de un tipo concreto, beneficiando a cualquier backend futuro.

### D4: `ChatOpenAI` desde LangChain para la capa de construcción de grafos

**Decisión**: El branch `"vllm"` en `init_langchain_model()` devuelve `ChatOpenAI(api_key=..., base_url=..., model=..., ...)`. `ChatOpenAI` usa internamente el SDK de OpenAI, así que es compatible con la API OpenAI-compatible de vLLM. Al ser `isinstance(client, ChatOpenAI) == True`, el código existente de NER/OpenIE que chequea por `ChatOpenAI` entraría en el camino de JSON mode — pero con la bandera `json_mode=False` en el config, el código usará el camino de `extract_json_dict` (fallback), que es el comportamiento correcto para servidores sin guided decoding.

### D5: Parámetros vía YAML config, no env vars

**Alternativas consideradas:**
- Env vars (`VLLM_API_BASE`, `VLLM_API_KEY`) — patrón usado por `OPENAI_API_KEY`, `GOOGLE_API_KEY`, etc. Pero `Qwen3TextEmbModel` ya usa parámetros directos para vLLM.
- Parámetros en YAML — más explícito, permite múltiples perfiles (distintos servidores para distintos workflows), no requiere modificar `.env`.

**Decisión**: Parámetros en YAML config para todos los valores de conexión (`base_url`, `api_key`). Esto es consistente con cómo `Qwen3TextEmbModel` recibe `api_base` y `api_key`.

### D6: Auto-detección de `max_model_len` desde `/v1/models`

**Decisión**: Si `maximun_token` no se especifica en el config (None), `VLLMModel` consulta `GET /v1/models` al init y extrae `max_model_len` del modelo. Fallback: 4096 si la consulta falla. Esto evita que el usuario tenga que conocer y hardcodear el contexto máximo del modelo.

**Nota**: Se mantiene el nombre `maximun_token` (con typo) por consistencia con el resto de implementaciones del proyecto.

## Risks / Trade-offs

- **[Riesgo] JSON mode en NER/OpenIE con modelos que no soportan guided decoding**: Si un usuario configura `json_mode: true` con un servidor vLLM sin guided decoding, las llamadas fallarán. → **Mitigación**: El default para `json_mode` es `true` (backward-compatible con OpenAI), pero la documentación y los configs de ejemplo para vLLM usarán `false`. El error de vLLM será capturado por los try/except existentes en NER/OpenIE.

- **[Riesgo] `/tokenize` no disponible en versiones antiguas de vLLM**: El endpoint `/tokenize` se añadió en una versión relativamente reciente. → **Mitigación**: Incluir fallback a `AutoTokenizer` si `/tokenize` devuelve 404.

- **[Riesgo] `max_model_len` no expuesto en `/v1/models`**: Algunas versiones de vLLM pueden no incluir este campo. → **Mitigación**: Fallback a 4096, y el usuario siempre puede especificar `maximun_token` explícitamente en el config.

- **[Trade-off] `ChatOpenAI` con `base_url` para vLLM**: LangChain puede añadir headers o comportamientos inesperados al usar `ChatOpenAI` con un base_url no-OpenAI. → **Mitigación**: vLLM implementa la API OpenAI de forma compatible; `ChatOpenAI` simplemente reenvía requests al base_url configurado. Probado en la práctica con otros proyectos.

## Open Questions

- ¿Debería `VLLMModel` soportar `AsyncOpenAI` para llamadas asíncronas? Los workflows actuales usan `ThreadPoolExecutor` con llamadas síncronas, así que no es necesario en esta iteración.
- ¿Deberían los Hydra configs de vLLM ser defaults compuestos (como los configs actuales de QA) o overrides manuales? Dado que los defaults actuales apuntan a `gpt-4o-mini`, los configs vLLM serán archivos separados para que el usuario los use como `--config-name`.
