import hashlib
import os

import requests
import torch
import torch.nn.functional as F
from openai import OpenAI
from tqdm import tqdm

from .base_model import BaseELModel


class VLLMELModel(BaseELModel):
    """Entity Linking Model that delegates embeddings to a remote vLLM server.

    Uses the OpenAI-compatible ``/v1/embeddings`` endpoint so the ``vllm``
    package is not required locally.

    Args:
        model_name (str): Name of the embedding model served by vLLM.
        api_base (str): Base URL of the vLLM server (e.g. ``http://localhost:8000/v1``).
        api_key (str, optional): API key for the server. Defaults to ``"EMPTY"``.
        root (str, optional): Root directory for embedding cache. Defaults to ``"tmp"``.
        use_cache (bool, optional): Whether to cache entity embeddings on disk. Defaults to ``True``.
        normalize (bool, optional): Whether to apply L2 normalization to embeddings. Defaults to ``True``.
        batch_size (int, optional): Number of texts per request to the server. Defaults to ``32``.
        query_instruct (str | None, optional): Prefix prepended to NER query texts. Defaults to ``None``.
        passage_instruct (str | None, optional): Prefix prepended to entity passage texts. Defaults to ``None``.
        vllm_timeout (int, optional): Request timeout in seconds. Defaults to ``600``.
    """

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str = "EMPTY",
        root: str = "tmp",
        use_cache: bool = True,
        normalize: bool = True,
        batch_size: int = 32,
        query_instruct: str | None = None,
        passage_instruct: str | None = None,
        vllm_timeout: int = 600,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.use_cache = use_cache
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct
        self.vllm_timeout = vllm_timeout

        self.root = os.path.join(root, f"{model_name.replace('/', '_')}_vllm_cache")
        if use_cache and not os.path.exists(self.root):
            os.makedirs(self.root)

        if not self._is_api_available():
            raise RuntimeError(
                f"vLLM server not available at {api_base}. "
                "Ensure the server is running before instantiating VLLMELModel."
            )

        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def _is_api_available(self) -> bool:
        try:
            health_url = self.api_base.replace("/v1", "/health")
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _add_instruct(self, instruct: str | None, texts: list[str]) -> list[str]:
        if instruct is None:
            return texts
        return [f"{instruct}{t}" for t in texts]

    def _embed(self, texts: list[str], show_progress_bar: bool = True) -> torch.Tensor:
        all_embeddings: list[list[float]] = []
        for i in tqdm(
            range(0, len(texts), self.batch_size), disable=not show_progress_bar
        ):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                timeout=self.vllm_timeout,
            )
            all_embeddings.extend(data.embedding for data in response.data)
        if not all_embeddings:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.tensor(all_embeddings, dtype=torch.float32)

    def index(self, entity_list: list) -> None:
        """Build the entity index from ``entity_list``.

        Embeddings are fetched from the vLLM server in batches and optionally
        cached on disk using an MD5 fingerprint of the entity list.
        """
        self.entity_list = entity_list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        cache_file = os.path.join(self.root, f"{fingerprint}.pt")

        if self.use_cache and os.path.exists(cache_file):
            self.entity_embeddings = torch.load(
                cache_file, map_location="cpu", weights_only=True
            )
            return

        texts = self._add_instruct(self.passage_instruct, entity_list)
        embeddings = self._embed(texts, show_progress_bar=True)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        self.entity_embeddings = embeddings

        if self.use_cache:
            torch.save(self.entity_embeddings, cache_file)

    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """Link named entities to indexed entities and return the top-k matches.

        Args:
            ner_entity_list (list): Named entities to link.
            topk (int): Number of candidates to return per entity.

        Returns:
            dict: Maps each NER entity to a list of dicts with keys
                ``entity``, ``score``, and ``norm_score``.
        """
        if not ner_entity_list or self.entity_embeddings.shape[0] == 0:
            return {}

        texts = self._add_instruct(self.query_instruct, ner_entity_list)
        ner_embeddings = self._embed(texts, show_progress_bar=False)
        if self.normalize:
            ner_embeddings = F.normalize(ner_embeddings, p=2, dim=-1)

        topk = min(topk, self.entity_embeddings.shape[0])
        scores = ner_embeddings @ self.entity_embeddings.mT
        top_k_scores, top_k_indices = torch.topk(scores, topk, dim=-1)

        linked_entity_dict: dict[str, list] = {}
        for i, ner_entity in enumerate(ner_entity_list):
            sorted_scores = top_k_scores[i]
            sorted_indices = top_k_indices[i]
            max_score = sorted_scores[0].item()
            linked_entity_dict[ner_entity] = [
                {
                    "entity": self.entity_list[idx],
                    "score": score.item(),
                    "norm_score": score.item() / max_score,
                }
                for score, idx in zip(sorted_scores, sorted_indices)
            ]
        return linked_entity_dict
