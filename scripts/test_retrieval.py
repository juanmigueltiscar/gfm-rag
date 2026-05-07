#!/usr/bin/env python3
"""Test GFM-RAG retrieval with a local vLLM endpoint and a toy dataset."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

os.environ["HF_HOME"] = os.path.expanduser(
    os.environ.get("HF_HOME", "~/.hf_cache_user")
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def ensure_data(data_dir: str = "data", data_name: str = "toy_raw") -> Path:
    raw_dir = Path(data_dir) / data_name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    docs_path = raw_dir / "documents.json"
    if not docs_path.exists():
        import json

        docs = {
            "France": "France is a country in Western Europe. Paris is its capital.",
            "Paris": "Paris is the capital and most populous city of France.",
            "Emmanuel Macron": "Emmanuel Macron has served as president of France since 2017.",
        }
        docs_path.write_text(json.dumps(docs, indent=2))
        logger.info("Created %s", docs_path)

    test_path = raw_dir / "test.json"
    if not test_path.exists():
        import json

        test_data = [
            {
                "id": "toy-1",
                "question": "Who is the president of France?",
                "answer": "Emmanuel Macron",
                "answer_aliases": ["Macron"],
                "supporting_documents": ["France", "Emmanuel Macron"],
            }
        ]
        test_path.write_text(json.dumps(test_data, indent=2))
        logger.info("Created %s", test_path)

    return raw_dir


def check_vllm(base_url: str) -> bool:
    import requests

    health_url = base_url.rstrip("/").replace("/v1", "") + "/health"
    try:
        r = requests.get(health_url, timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def main() -> None:
    vllm_url = os.getenv("VLLM_BASE_URL")
    vllm_api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    vllm_model = os.getenv("VLLM_MODEL")
    vllm_embed_url = os.getenv("VLLM_EMBED_BASE_URL")
    vllm_embed_model = os.getenv("VLLM_EMBED_MODEL")
    hf_token = os.getenv("HF_TOKEN")

    if not vllm_model:
        logger.critical(
            "VLLM_MODEL is not set in .env. Set it to the model name served by your vLLM server."
        )
        return

    if not hf_token:
        logger.critical(
            "HF_TOKEN is not set in .env. It is required to download gated models from HuggingFace."
        )
        return

    if not vllm_url:
        logger.critical("VLLM_BASE_URL is not set in .env.")
        return

    if not vllm_embed_url or not vllm_embed_model:
        logger.critical("VLLM_EMBED_BASE_URL and VLLM_EMBED_MODEL must be set in .env.")
        return

    logger.info("Checking vLLM LLM health at %s ...", vllm_url)
    if not check_vllm(vllm_url):
        logger.critical(
            "vLLM LLM server is not reachable. Start it first, e.g.:\n"
            "  vllm serve %s --port 8082",
            vllm_model,
        )
        return
    logger.info("vLLM LLM server is healthy")

    logger.info("Checking vLLM embed health at %s ...", vllm_embed_url)
    if not check_vllm(vllm_embed_url):
        logger.critical(
            "vLLM embed server is not reachable. Start it first, e.g.:\n"
            "  vllm serve %s --port 8000",
            vllm_embed_model,
        )
        return
    logger.info("vLLM embed server is healthy")

    data_dir = "data"
    data_name = "toy_raw"
    model_path = "rmanluo/G-reasoner-34M"

    ensure_data(data_dir, data_name)

    from gfmrag.graph_index_construction.entity_linking_model import VLLMELModel
    from gfmrag.graph_index_construction.graph_constructors import KGConstructor
    from gfmrag.graph_index_construction.ner_model import LLMNERModel
    from gfmrag.graph_index_construction.openie_model import LLMOPENIEModel

    logger.info("Initializing vLLM-based NER model (%s)...", vllm_model)
    ner_model = LLMNERModel(
        llm_api="vllm",
        model_name=vllm_model,
        base_url=vllm_url,
        api_key=vllm_api_key,
        max_tokens=int(os.getenv("VLLM_MAX_TOKENS", "512")),
        json_mode=True,
    )
    logger.info("NER model ready")

    logger.info("Initializing vLLM-based OpenIE model (%s)...", vllm_model)
    openie_model = LLMOPENIEModel(
        llm_api="vllm",
        model_name=vllm_model,
        base_url=vllm_url,
        api_key=vllm_api_key,
        max_ner_tokens=int(os.getenv("VLLM_MAX_TOKENS", "512")),
        max_triples_tokens=int(os.getenv("VLLM_MAX_TRIPLES_TOKENS", "512")),
        json_mode=True,
    )
    logger.info("OpenIE model ready")

    logger.info(
        "Initializing VLLMELModel for graph construction (%s)...", vllm_embed_model
    )
    el_for_graph = VLLMELModel(
        model_name=vllm_embed_model,
        api_base=vllm_embed_url,
        api_key=vllm_api_key,
    )
    logger.info("VLLMELModel (graph) ready")

    graph_constructor = KGConstructor(
        open_ie_model=openie_model,
        el_model=el_for_graph,
        num_processes=1,
        cosine_sim_edges=True,
        threshold=0.8,
        max_sim_neighbors=100,
        add_title=True,
    )
    logger.info("KGConstructor ready")

    logger.info(
        "Initializing VLLMELModel for query-time entity linking (%s)...",
        vllm_embed_model,
    )
    el_for_query = VLLMELModel(
        model_name=vllm_embed_model,
        api_base=vllm_embed_url,
        api_key=vllm_api_key,
    )
    logger.info("VLLMELModel (query) ready")

    from gfmrag import GFMRetriever

    text_emb_model_cfgs = {
        "_target_": "gfmrag.text_emb_models.Qwen3TextEmbModel",
        "text_emb_model_name": vllm_embed_model,
        "api_base": vllm_embed_url,
        "api_key": vllm_api_key,
        "normalize": True,
        "batch_size": 32,
        "query_instruct": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        "passage_instruct": None,
        "truncate_dim": 1024,
    }

    logger.info(
        "Building GFMRetriever from index (this may take a while on first run)..."
    )
    retriever = GFMRetriever.from_index(
        data_dir=data_dir,
        data_name=data_name,
        model_path=model_path,
        ner_model=ner_model,
        el_model=el_for_query,
        graph_constructor=graph_constructor,
        text_emb_model_cfgs=text_emb_model_cfgs,
    )
    logger.info("GFMRetriever ready")

    query = "Who is the president of France?"
    logger.info("Running single-query retrieval for: %s", query)
    results = retriever.retrieve(query, top_k=5)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Single query: {query}")
    print(sep)
    for rank, item in enumerate(results["document"], 1):
        print(f"  #{rank}  {item['id']:25s}  score={item['score']:.4f}")
    print(sep)

    queries = [
        "Who is the president of France?",
        "What is the capital of France?",
        "Tell me about Emmanuel Macron.",
    ]
    logger.info("Running batch retrieval for %d queries", len(queries))
    batch_results = retriever.retrieve(queries, top_k=5, max_batch_size=2)

    for idx, (q, res) in enumerate(zip(queries, batch_results, strict=False)):
        print(f"\n{sep}")
        print(f"Batch query #{idx + 1}: {q}")
        print(sep)
        for rank, item in enumerate(res["document"], 1):
            print(f"  #{rank}  {item['id']:25s}  score={item['score']:.4f}")
        print(sep)


if __name__ == "__main__":
    main()
