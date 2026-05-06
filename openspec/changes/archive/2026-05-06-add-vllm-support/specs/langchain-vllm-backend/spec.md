## ADDED Requirements

### Requirement: vLLM branch in init_langchain_model factory
`init_langchain_model()` SHALL support a new LLM identifier `"vllm"` that returns a `ChatOpenAI` instance configured with a custom `base_url` pointing to a vLLM server. The factory SHALL accept `base_url` and `api_key` as keyword arguments.

#### Scenario: Factory creates ChatOpenAI for vLLM
- **WHEN** `init_langchain_model(llm="vllm", model_name="meta-llama/Meta-Llama-3-8B", base_url="http://localhost:8000/v1", api_key="EMPTY")` is called
- **THEN** it returns a `ChatOpenAI` instance with `model="meta-llama/Meta-Llama-3-8B"`, `openai_api_base="http://localhost:8000/v1"` (or equivalent), and `openai_api_key="EMPTY"`

#### Scenario: Factory preserves all existing parameters
- **WHEN** `init_langchain_model(llm="vllm", ...)` is called with `temperature`, `max_retries`, `timeout` parameters
- **THEN** the returned `ChatOpenAI` instance is configured with those values

#### Scenario: Unknown LLM still raises error
- **WHEN** `init_langchain_model(llm="unknown_backend", ...)` is called
- **THEN** a `NotImplementedError` is raised

### Requirement: NER model accepts base_url and api_key for vLLM
`LLMNERModel.__init__()` SHALL accept `base_url` and `api_key` as optional parameters. When `llm_api="vllm"`, these parameters SHALL be forwarded to `init_langchain_model()`.

#### Scenario: NER model with vLLM backend
- **WHEN** `LLMNERModel(llm_api="vllm", model_name="meta-llama/Meta-Llama-3-8B", base_url="http://localhost:8000/v1", api_key="EMPTY")` is instantiated
- **THEN** `self.client` is a `ChatOpenAI` instance configured for the vLLM server

#### Scenario: NER model with OpenAI backend unchanged
- **WHEN** `LLMNERModel(llm_api="openai", model_name="gpt-4o-mini")` is instantiated without `base_url` or `api_key`
- **THEN** `self.client` is a `ChatOpenAI` instance configured for the default OpenAI API

### Requirement: OpenIE model accepts base_url and api_key for vLLM
`LLMOPENIEModel.__init__()` SHALL accept `base_url` and `api_key` as optional parameters. When `llm_api="vllm"`, these parameters SHALL be forwarded to `init_langchain_model()`.

#### Scenario: OpenIE model with vLLM backend
- **WHEN** `LLMOPENIEModel(llm_api="vllm", model_name="meta-llama/Meta-Llama-3-8B", base_url="http://localhost:8000/v1", api_key="EMPTY")` is instantiated
- **THEN** `self.client` is a `ChatOpenAI` instance configured for the vLLM server

#### Scenario: OpenIE model with OpenAI backend unchanged
- **WHEN** `LLMOPENIEModel(llm_api="openai", model_name="gpt-4o-mini")` is instantiated without `base_url` or `api_key`
- **THEN** `self.client` is a `ChatOpenAI` instance configured for the default OpenAI API

### Requirement: Hydra config for NER with vLLM
The NER model Hydra config SHALL support `llm_api: vllm` with `base_url` and `api_key` fields.

#### Scenario: NER instantiation from vLLM Hydra config
- **WHEN** a Hydra config specifies `{_target_: gfmrag.graph_index_construction.ner_model.LLMNERModel, llm_api: vllm, model_name: "meta-llama/Meta-Llama-3-8B", base_url: "http://localhost:8000/v1", api_key: "EMPTY"}`
- **THEN** `instantiate(cfg)` returns a functional `LLMNERModel` instance using vLLM

### Requirement: Hydra config for OpenIE with vLLM
The OpenIE model Hydra config SHALL support `llm_api: vllm` with `base_url` and `api_key` fields.

#### Scenario: OpenIE instantiation from vLLM Hydra config
- **WHEN** a Hydra config specifies `{_target_: gfmrag.graph_index_construction.openie_model.LLMOPENIEModel, llm_api: vllm, model_name: "meta-llama/Meta-Llama-3-8B", base_url: "http://localhost:8000/v1", api_key: "EMPTY"}`
- **THEN** `instantiate(cfg)` returns a functional `LLMOPENIEModel` instance using vLLM
