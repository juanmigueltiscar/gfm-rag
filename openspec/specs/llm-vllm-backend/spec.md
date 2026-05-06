## ADDED Requirements

### Requirement: VLLMModel initialization with health check
`VLLMModel` SHALL verify the vLLM server is reachable during `__init__` by performing a health check against `{base_url}/health`. If the server is unreachable or returns a non-200 status, the constructor SHALL raise a `RuntimeError` with a descriptive message.

#### Scenario: Server reachable at init time
- **WHEN** `VLLMModel` is instantiated with a valid `base_url` pointing to a running vLLM server
- **THEN** the health check succeeds and `self.client` is initialized as an `OpenAI` client with the configured `base_url` and `api_key`

#### Scenario: Server unreachable at init time
- **WHEN** `VLLMModel` is instantiated with a `base_url` pointing to an unreachable server
- **THEN** a `RuntimeError` is raised with message "vLLM API is not available"

### Requirement: Auto-detection of maximum token limit
`VLLMModel` SHALL query `GET {base_url}/v1/models` during initialization to auto-detect the model's `max_model_len`. If the field is not present or the request fails, it SHALL fall back to 4096. If the user provides `maximun_token` explicitly in the constructor, the auto-detection SHALL be skipped.

#### Scenario: Token limit auto-detected from server
- **WHEN** `VLLMModel` is instantiated with `maximun_token=None` and the server returns a model with `max_model_len: 32768`
- **THEN** `self.maximun_token` is set to 32768

#### Scenario: Token limit explicitly provided
- **WHEN** `VLLMModel` is instantiated with `maximun_token=8192`
- **THEN** `self.maximun_token` is set to 8192 regardless of the server's model config

#### Scenario: Token limit auto-detection fails
- **WHEN** `VLLMModel` is instantiated with `maximun_token=None` and the `/v1/models` endpoint returns no `max_model_len` field
- **THEN** `self.maximun_token` is set to 4096 as fallback

### Requirement: Tokenization via vLLM /tokenize endpoint
`VLLMModel.token_len()` SHALL compute the token count by sending a POST request to `{base_url}/tokenize` with the model name and the text. If the `/tokenize` endpoint is not available (404), it SHALL fall back to using `AutoTokenizer.from_pretrained()` as a secondary method.

#### Scenario: Successful tokenization
- **WHEN** `token_len("Hello world")` is called and the server supports `/tokenize`
- **THEN** the method returns the integer token count as reported by the vLLM server

#### Scenario: /tokenize endpoint not available
- **WHEN** `token_len("Hello world")` is called and `/tokenize` returns 404
- **THEN** the method falls back to `AutoTokenizer.from_pretrained(model_name_or_path)` and returns the token count

### Requirement: Text generation with retry and truncation
`VLLMModel.generate_sentence()` SHALL send messages to the vLLM server via `client.chat.completions.create()` using the configured `model_name`, `temperature`, and `timeout`. It SHALL truncate the input if it exceeds `maximun_token`. It SHALL retry up to `retry` times with a 30-second delay between attempts on failure, returning the exception if all retries are exhausted.

#### Scenario: Successful generation
- **WHEN** `generate_sentence("What is the capital of France?")` is called with a functioning server
- **THEN** the method returns the stripped text response from the model

#### Scenario: Input exceeds token limit
- **WHEN** `generate_sentence()` is called with an input longer than `maximun_token`
- **THEN** the input is truncated to `maximun_token` tokens before sending

#### Scenario: Generation fails and retries
- **WHEN** `generate_sentence()` fails on the first attempt due to a transient server error
- **THEN** the method retries after a 30-second delay, up to `retry` times

#### Scenario: All retries exhausted
- **WHEN** `generate_sentence()` fails on every retry attempt
- **THEN** the method returns the last `Exception` instance

### Requirement: Hydra instantiation support
`VLLMModel` SHALL be importable from `gfmrag.llms` and instantiatable via Hydra's `instantiate()` with `_target_: gfmrag.llms.VLLMModel` in YAML configuration.

#### Scenario: Instantiation via Hydra config
- **WHEN** a Hydra config specifies `llm: {_target_: gfmrag.llms.VLLMModel, model_name_or_path: "meta-llama/Meta-Llama-3-8B", base_url: "http://localhost:8000/v1"}`
- **THEN** `instantiate(cfg.llm)` returns a functional `VLLMModel` instance

### Requirement: Configurable via YAML parameters
All connection parameters of `VLLMModel` (`base_url`, `api_key`, `retry`, `timeout`, `maximun_token`, `temperature`) SHALL be configurable through Hydra YAML config files, not through environment variables.

#### Scenario: QA workflow with vLLM config
- **WHEN** a QA inference config uses `_target_: gfmrag.llms.VLLMModel` with all parameters specified in YAML
- **THEN** the workflow runs using the vLLM-backed model without requiring any environment variables beyond what vLLM needs
