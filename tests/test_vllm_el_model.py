import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

ENTITY_LIST = ["Paris", "London", "Berlin"]
NER_LIST = ["paris city"]

# Fixed embeddings: Paris ≈ [1,0,0], London ≈ [0,1,0], Berlin ≈ [0,0,1]
_EMBEDDINGS = {
    "Paris": [1.0, 0.0, 0.0],
    "London": [0.0, 1.0, 0.0],
    "Berlin": [0.0, 0.0, 1.0],
    "paris city": [0.9, 0.1, 0.0],
}


def _make_embedding_response(texts: list[str]) -> MagicMock:
    response = MagicMock()
    response.data = [
        SimpleNamespace(embedding=_EMBEDDINGS.get(t, [0.5, 0.5, 0.0])) for t in texts
    ]
    return response


def _patch_health_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    monkeypatch.setattr(
        "gfmrag.graph_index_construction.entity_linking_model.vllm_el_model.requests.get",
        lambda *a, **kw: mock_resp,
    )


def _make_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    _patch_health_ok(monkeypatch)
    from gfmrag.graph_index_construction.entity_linking_model.vllm_el_model import (
        VLLMELModel,
    )

    with patch(
        "gfmrag.graph_index_construction.entity_linking_model.vllm_el_model.OpenAI"
    ) as mock_open_ai:
        client = mock_open_ai.return_value
        client.embeddings.create.side_effect = (
            lambda model, input, **kw: _make_embedding_response(input)
        )
        kw = {"normalize": False, **kwargs}
        model = VLLMELModel(
            model_name="test-model",
            api_base="http://localhost:8000/v1",
            root=str(tmp_path),
            **kw,
        )
        model.client = client
    return model


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


def test_init_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(tmp_path, monkeypatch)
    assert model.model_name == "test-model"
    assert model.client is not None


def test_init_server_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests as _requests

    monkeypatch.setattr(
        "gfmrag.graph_index_construction.entity_linking_model.vllm_el_model.requests.get",
        MagicMock(side_effect=_requests.ConnectionError("connection refused")),
    )
    from gfmrag.graph_index_construction.entity_linking_model.vllm_el_model import (
        VLLMELModel,
    )

    with pytest.raises(RuntimeError, match="vLLM server not available"):
        VLLMELModel(
            model_name="test-model",
            api_base="http://localhost:9999/v1",
            root=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# index() tests
# ---------------------------------------------------------------------------


def test_index_calls_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(tmp_path, monkeypatch, use_cache=False)
    model.index(ENTITY_LIST)

    assert hasattr(model, "entity_embeddings")
    assert model.entity_embeddings.shape == (3, 3)
    assert model.entity_list == ENTITY_LIST


def test_index_saves_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(tmp_path, monkeypatch, use_cache=True)
    model.index(ENTITY_LIST)

    fingerprint = hashlib.md5("".join(ENTITY_LIST).encode()).hexdigest()
    cache_file = os.path.join(model.root, f"{fingerprint}.pt")
    assert os.path.exists(cache_file)


def test_index_loads_from_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_model(tmp_path, monkeypatch, use_cache=True)
    model.index(ENTITY_LIST)
    call_count_after_first = model.client.embeddings.create.call_count

    # Second index call should load from cache without hitting the server
    model2 = _make_model(tmp_path, monkeypatch, use_cache=True)
    model2.index(ENTITY_LIST)
    assert model2.client.embeddings.create.call_count == 0

    # But the embeddings should be identical
    assert torch.allclose(model.entity_embeddings, model2.entity_embeddings)
    _ = call_count_after_first  # used implicitly above


def test_index_normalize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(tmp_path, monkeypatch, use_cache=False, normalize=True)
    model.index(ENTITY_LIST)

    norms = model.entity_embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# __call__() tests
# ---------------------------------------------------------------------------


def test_call_returns_correct_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_model(tmp_path, monkeypatch, use_cache=False)
    model.index(ENTITY_LIST)
    result = model(NER_LIST, topk=2)

    assert "paris city" in result
    assert len(result["paris city"]) == 2
    first = result["paris city"][0]
    assert "entity" in first
    assert "score" in first
    assert "norm_score" in first


def test_call_norm_score_top_is_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_model(tmp_path, monkeypatch, use_cache=False)
    model.index(ENTITY_LIST)
    result = model(NER_LIST, topk=2)
    assert result["paris city"][0]["norm_score"] == pytest.approx(1.0)


def test_call_top_result_is_paris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_model(tmp_path, monkeypatch, use_cache=False)
    model.index(ENTITY_LIST)
    result = model(["Paris"], topk=1)
    assert result["Paris"][0]["entity"] == "Paris"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_call_empty_entity_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_model(tmp_path, monkeypatch, use_cache=False)
    model.index([])
    assert model.entity_embeddings.shape == (0, 0)
    result = model(["Paris"], topk=10)
    assert result == {}


def test_call_topk_clamped_to_entity_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_model(tmp_path, monkeypatch, use_cache=False)
    model.index(ENTITY_LIST)  # 3 entities
    result = model(["Paris"], topk=100)
    assert len(result["Paris"]) == len(ENTITY_LIST)


# ---------------------------------------------------------------------------
# _add_instruct tests
# ---------------------------------------------------------------------------


def test_add_instruct_prepends_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_model(tmp_path, monkeypatch)
    out = model._add_instruct("Query: ", ["Paris", "London"])
    assert out == ["Query: Paris", "Query: London"]


def test_add_instruct_none_returns_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_model(tmp_path, monkeypatch)
    texts = ["Paris", "London"]
    assert model._add_instruct(None, texts) is texts
