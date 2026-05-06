## ADDED Requirements

### Requirement: json_mode flag in LLMNERModel
`LLMNERModel.__init__()` SHALL accept an optional `json_mode: bool = True` parameter. When `json_mode=True`, the model SHALL use structured JSON mode in LLM calls (e.g., `response_format={"type": "json_object"}`). When `json_mode=False`, the model SHALL use `extract_json_dict()` to parse unstructured responses. The flag SHALL replace `isinstance(self.client, ChatOpenAI)` checks throughout the class.

#### Scenario: json_mode=True uses structured output
- **WHEN** `LLMNERModel(json_mode=True)` processes text and the client supports `response_format`
- **THEN** the model calls the client with `response_format={"type": "json_object"}`

#### Scenario: json_mode=False uses extract_json_dict
- **WHEN** `LLMNERModel(json_mode=False)` processes text
- **THEN** the model calls the client without `response_format` and parses the response using `extract_json_dict()`

#### Scenario: Default json_mode=True preserves backward compatibility
- **WHEN** `LLMNERModel()` is instantiated without specifying `json_mode`
- **THEN** `self.json_mode` is `True` and behavior is identical to before the change

### Requirement: json_mode flag in LLMOPENIEModel
`LLMOPENIEModel.__init__()` SHALL accept an optional `json_mode: bool = True` parameter. When `json_mode=True`, the model SHALL use structured JSON mode in LLM calls. When `json_mode=False`, the model SHALL use `extract_json_dict()` to parse unstructured responses. The flag SHALL replace `isinstance(self.client, ChatOpenAI)` checks throughout the class.

#### Scenario: json_mode=True uses structured output
- **WHEN** `LLMOPENIEModel(json_mode=True)` processes text for NER and OpenIE and the client supports `response_format`
- **THEN** the model calls the client with `response_format={"type": "json_object"}` for both NER and triple extraction

#### Scenario: json_mode=False uses extract_json_dict
- **WHEN** `LLMOPENIEModel(json_mode=False)` processes text for NER and OpenIE
- **THEN** the model calls the client without `response_format` and parses responses using `extract_json_dict()`

#### Scenario: Default json_mode=True preserves backward compatibility
- **WHEN** `LLMOPENIEModel()` is instantiated without specifying `json_mode`
- **THEN** `self.json_mode` is `True` and behavior is identical to before the change

### Requirement: json_mode in Hydra configs for NER
The NER model Hydra config SHALL include `json_mode: true` as a configurable field with a default value of `true`.

#### Scenario: NER config with json_mode=true
- **WHEN** a Hydra config for NER specifies `json_mode: true`
- **THEN** `instantiate(cfg)` creates an `LLMNERModel` with `json_mode=True`

#### Scenario: NER config with json_mode=false
- **WHEN** a Hydra config for NER specifies `json_mode: false`
- **THEN** `instantiate(cfg)` creates an `LLMNERModel` with `json_mode=False`

### Requirement: json_mode in Hydra configs for OpenIE
The OpenIE model Hydra config SHALL include `json_mode: true` as a configurable field with a default value of `true`.

#### Scenario: OpenIE config with json_mode=true
- **WHEN** a Hydra config for OpenIE specifies `json_mode: true`
- **THEN** `instantiate(cfg)` creates an `LLMOPENIEModel` with `json_mode=True`

#### Scenario: OpenIE config with json_mode=false
- **WHEN** a Hydra config for OpenIE specifies `json_mode: false`
- **THEN** `instantiate(cfg)` creates an `LLMOPENIEModel` with `json_mode=False`
