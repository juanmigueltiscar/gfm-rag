import asyncio
import logging
import threading

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gfmrag import utils
from gfmrag.datasets import QADataset
from gfmrag.doc_rankers import BaseDocRanker
from gfmrag.kg_construction.entity_linking_model import BaseELModel
from gfmrag.kg_construction.ner_model import BaseNERModel
from gfmrag.models import GNNRetriever
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.ultra import query_utils
from gfmrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)


class GFMRetriever:
    """Graph Foundation Model (GFM) Retriever for document retrieval.

    This class implements a document retrieval system that combines named entity recognition,
    entity linking, graph neural networks, and document ranking to retrieve relevant documents
    based on a query.

    Attributes:
        qa_data (QADataset): Dataset containing the knowledge graph, documents and mappings
        graph (torch.Tensor): Knowledge graph structure
        text_emb_model (BaseTextEmbModel): Model for text embedding
        ner_model (BaseNERModel): Named Entity Recognition model
        el_model (BaseELModel): Entity Linking model
        graph_retriever (GNNRetriever): Graph Neural Network based retriever
        doc_ranker (BaseDocRanker): Document ranking model
        doc_retriever (DocumentRetriever): Document retrieval utility
        device (torch.device): Device to run computations on
        num_nodes (int): Number of nodes in the knowledge graph
        entities_weight (torch.Tensor | None): Optional weights for entities

    Examples:
        >>> retriever = GFMRetriever.from_config(cfg)
        >>> docs = retriever.retrieve("Who is the president of France?", top_k=5)
    """

    def __init__(
        self,
        qa_data: QADataset,
        text_emb_model: BaseTextEmbModel,
        ner_model: BaseNERModel,
        el_model: BaseELModel,
        graph_retriever: GNNRetriever,
        doc_ranker: BaseDocRanker,
        doc_retriever: utils.DocumentRetriever,
        entities_weight: torch.Tensor | None,
        device: torch.device,
    ) -> None:
        self.qa_data = qa_data
        self.graph = qa_data.kg
        self.text_emb_model = text_emb_model
        self.ner_model = ner_model
        self.el_model = el_model
        self.graph_retriever = graph_retriever
        self.doc_ranker = doc_ranker
        self.doc_retriever = doc_retriever
        self.device = device
        self.num_nodes = self.graph.num_nodes
        self.entities_weight = entities_weight

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int) -> tuple[list[dict], list]:
        """
        Retrieve documents from the corpus based on the given query.

        1. Prepares the query input for the graph retriever
        2. Executes the graph retriever forward pass to get entity predictions
        3. Ranks documents based on entity predictions
        4. Retrieves the top-k supporting documents

        Args:
            query (str): input query
            top_k (int): number of documents to retrieve

        Returns:
            list[dict]: A list of retrieved documents, where each document is represented as a dictionary
                        containing document metadata and content
        """
        
        
        # Prepare input for deep graph retriever
        diccionarioLista =  self.prepare_input_for_graph_retriever(query)
        diccionario = diccionarioLista[0]
        entidades = diccionarioLista[1] 
        #graph_retriever_input = self.prepare_input_for_graph_retriever(query)[0]
        graph_retriever_input = diccionario
        graph_retriever_input = query_utils.cuda(
            graph_retriever_input, device=self.device
        )

        # Graph retriever forward pass
        ent_pred = self.graph_retriever(
            self.graph, graph_retriever_input, entities_weight=self.entities_weight
        )
        doc_pred = self.doc_ranker(ent_pred)[0]  # Ent2docs mapping, batch size is 1

        # Retrieve the supporting documents
        retrieved_docs = self.doc_retriever(doc_pred.cpu(), top_k=top_k)

        return retrieved_docs, entidades

    def prepare_input_for_graph_retriever(self, query: str) -> tuple[dict, list]:
        """
        Prepare input for the graph retriever model by processing the query through entity detection, linking and embedding generation. The function performs the following steps:

        1. Detects entities in the query using NER model
        2. Links detected entities to knowledge graph entities
        3. Converts entities to node masks
        4. Generates question embeddings
        5. Combines embeddings and masks into input format

        Args:
            query (str): Input query text to process

        Returns:
            dict: Dictionary containing processed inputs with keys:

                - question_embeddings: Embedded representation of the query
                - question_entities_masks: Binary mask tensor indicating entity nodes (shape: 1 x num_nodes)

        Notes:
            - If no entities are detected in query, the full query is used for entity linking
            - Only linked entities that exist in qa_data.ent2id are included in masks
            - Entity masks and embeddings are formatted for graph retriever model input
        """

        # Prepare input for deep graph retriever
        mentioned_entities = self.ner_model(query)
        #ITC_MODIFICADO
        #print('------------------ Entitats mencionades')
        print(mentioned_entities)
        if len(mentioned_entities) == 0:
            logger.warning(
                "No mentioned entities found in the query. Use the query as is for entity linking."
            )
            mentioned_entities = [query]
        linked_entities = self.el_model(mentioned_entities, topk=1)
        entity_ids = [
            self.qa_data.ent2id[ent[0]["entity"]]
            for ent in linked_entities.values()
            if ent[0]["entity"] in self.qa_data.ent2id
        ]
        question_entities_masks = (
            entities_to_mask(entity_ids, self.num_nodes).unsqueeze(0).to(self.device)
        )  # 1 x num_nodes
        question_embedding = self.text_emb_model.encode(
            [query],
            is_query=True,
            show_progress_bar=False,
        )
        graph_retriever_input = {
            "question_embeddings": question_embedding,
            "question_entities_masks": question_entities_masks,
        }
        return graph_retriever_input, mentioned_entities

    @torch.no_grad()
    def retrieve_batch(
        self,
        queries: list[str],
        top_k: int,
        max_parallel_ner: int = 20,
    ) -> list[tuple[list[dict], list]]:
        """Procesa múltiples queries en batch, optimizando el uso de GPU y la red.

        Estrategia de optimización:
        - NER: concurrente con asyncio.gather() (en chunks de max_parallel_ner)
        - Text embedding: una sola llamada encode(all_queries)
        - EL + masks: secuencial pero local (< 100ms total)
        - GNN forward: un solo forward pass para todo el batch
        - DocRanker: un solo matmul para todo el batch
        - DocumentRetriever: bucle Python (CPU, rápido)

        Args:
            queries: Lista de queries a procesar.
            top_k: Número de documentos a recuperar por query.
            max_parallel_ner: Máximo de llamadas NER simultáneas (evita saturar el servidor).

        Returns:
            Lista de (retrieved_docs, mentioned_entities) en el mismo orden que queries.
        """
        # ── PASO 1: NER concurrente ─────────────────────────────────────────────
        async def _run_ner_batch() -> list[list[str]]:
            results: list[list[str]] = []
            for i in range(0, len(queries), max_parallel_ner):
                chunk = queries[i : i + max_parallel_ner]
                chunk_results = await asyncio.gather(
                    *[self.ner_model.acall(q) for q in chunk]
                )
                results.extend(chunk_results)
            return results

        # Compatibilidad con entornos que ya tienen event loop activo (ej: Jupyter)
        try:
            asyncio.get_running_loop()
            # Event loop ya activo → ejecutar en un thread con su propio loop
            result_holder: list[list[list[str]]] = []

            def _run_in_new_loop() -> None:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result_holder.append(new_loop.run_until_complete(_run_ner_batch()))
                finally:
                    new_loop.close()

            t = threading.Thread(target=_run_in_new_loop)
            t.start()
            t.join()
            mentioned_entities_list = result_holder[0]
        except RuntimeError:
            # No hay event loop activo → usar asyncio.run() directamente
            mentioned_entities_list = asyncio.run(_run_ner_batch())

        # Fallback: query sin entidades detectadas → usar la query completa
        for i, entities in enumerate(mentioned_entities_list):
            if not entities:
                logger.warning(
                    f"No entities found for query {i}, using full query for entity linking."
                )
                mentioned_entities_list[i] = [queries[i]]

        # ── PASO 2: Text embedding — una sola llamada para todo el batch ────────
        question_embeddings = self.text_emb_model.encode(
            queries, is_query=True, show_progress_bar=False
        )  # shape: (N, emb_dim)

        # ── PASO 3: EL + máscaras — secuencial pero local ──────────────────────
        masks = []
        for mentioned_entities in mentioned_entities_list:
            linked_entities = self.el_model(mentioned_entities, topk=1)
            entity_ids = [
                self.qa_data.ent2id[ent[0]["entity"]]
                for ent in linked_entities.values()
                if ent[0]["entity"] in self.qa_data.ent2id
            ]
            mask = entities_to_mask(entity_ids, self.num_nodes)
            masks.append(mask)

        # ── PASO 4: Stack → tensores batch ─────────────────────────────────────
        batch_input = {
            "question_embeddings": question_embeddings.to(self.device),
            "question_entities_masks": torch.stack(masks, dim=0).to(self.device),
        }

        # ── PASO 5: Un solo forward GNN para todo el batch ─────────────────────
        ent_pred = self.graph_retriever(
            self.graph, batch_input, entities_weight=self.entities_weight
        )  # shape: (N, n_entities)

        # ── PASO 6: Batch doc ranking — un solo matmul ─────────────────────────
        doc_preds = self.doc_ranker(ent_pred)  # shape: (N, n_docs)

        # ── PASO 7: DocumentRetriever por query (CPU, rápido) ──────────────────
        results = []
        for doc_pred, entities in zip(doc_preds, mentioned_entities_list):
            retrieved_docs = self.doc_retriever(doc_pred.cpu(), top_k=top_k)
            results.append((retrieved_docs, entities))

        return results

    @staticmethod
    def from_config(cfg: DictConfig) -> "GFMRetriever":
        """
        Constructs a GFMRetriever instance from a configuration dictionary.

        This factory method initializes all necessary components for the GFM retrieval system including:
        - Graph retrieval model
        - Question-answering dataset
        - Named Entity Recognition (NER) model
        - Entity Linking (EL) model
        - Document ranking and retrieval components
        - Text embedding model

        Args:
            cfg (DictConfig): Configuration dictionary containing settings for:

                - graph_retriever: Model path and NER/EL model configurations
                - dataset: Dataset parameters
                - Optional entity weight initialization flag

        Returns:
            GFMRetriever: Fully initialized retriever instance with all components loaded and
                          moved to appropriate device (CPU/GPU)

        Note:
            The configuration must contain valid paths and parameters for all required models
            and dataset components. Models are automatically moved to available device (CPU/GPU).
        """
        
        graph_retriever, model_config = utils.load_model_from_pretrained(
            cfg.graph_retriever.model_path
        )
        graph_retriever.eval()

        qa_data = QADataset(
            **cfg.dataset,
            text_emb_model_cfgs=OmegaConf.create(model_config["text_emb_model_config"]),
        )
        device = utils.get_device()
        graph_retriever = graph_retriever.to(device)

        qa_data.kg = qa_data.kg.to(device)
        ent2docs = qa_data.ent2docs.to(device)

        ner_model = instantiate(cfg.graph_retriever.ner_model)
        el_model = instantiate(cfg.graph_retriever.el_model)

        el_model.index(list(qa_data.ent2id.keys()))

        # Create doc ranker
        doc_ranker = instantiate(cfg.graph_retriever.doc_ranker, ent2doc=ent2docs)
        doc_retriever = utils.DocumentRetriever(qa_data.doc, qa_data.id2doc)

        text_emb_model = instantiate(
            OmegaConf.create(model_config["text_emb_model_config"])
        )

        entities_weight = None
        if cfg.graph_retriever.init_entities_weight:
            entities_weight = utils.get_entities_weight(ent2docs)

        return GFMRetriever(
            qa_data=qa_data,
            text_emb_model=text_emb_model,
            ner_model=ner_model,
            el_model=el_model,
            graph_retriever=graph_retriever,
            doc_ranker=doc_ranker,
            doc_retriever=doc_retriever,
            entities_weight=entities_weight,
            device=device,
        )
