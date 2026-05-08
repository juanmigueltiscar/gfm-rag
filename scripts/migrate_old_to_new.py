#!/usr/bin/env python3
"""
Migrate an old-format GFM-RAG dataset directory to the new graph format.

Reads from <old-dir>/:
  - processed/stage1/kg.txt                  (subject, relation, object triples)
  - processed/stage1/document2entities.json  (filename -> [entities])
  - raw/dataset_corpus.json                  (filename -> content)

Writes to <out-dir>/:
  - processed/stage1/nodes.csv      (name, type, attributes)
  - processed/stage1/edges.csv      (source, relation, target, attributes)
  - processed/stage1/relations.csv  (name, attributes)
  - raw/documents.json              (copy of dataset_corpus.json, renamed)
  - raw/<other files>               (all other files in <old-dir>/raw/, copied as-is)

Edge types generated:
  - Semantic (from kg.txt)
  - is_mentioned_in (entity -> document)
  - has section / contains section (markdown heading hierarchy)
  - equivalent (cosine-similarity synonymy via vLLM, optional)
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import unicodedata

import dotenv
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def processing_phrases(phrase: str) -> str:
    """Normalize a phrase: strip diacritics, lowercase, keep alphanumeric + spaces."""
    if isinstance(phrase, int):
        return str(phrase)
    nfd = unicodedata.normalize("NFD", phrase)
    no_diacritics = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return re.sub("[^A-Za-z0-9 ]", " ", no_diacritics.lower()).strip()


def parse_sections(content: str) -> list[tuple[int, str]]:
    """Return list of (heading_level, title) for ## and deeper headings."""
    sections = []
    for line in content.split("\n"):
        m = re.match(r"^(#{2,})\s+(.+?)\s*$", line)
        if m:
            sections.append((len(m.group(1)), m.group(2).strip()))
    return sections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="Migrate old GFM-RAG stage1 data to new CSV format"
    )
    parser.add_argument("--old-dir", default="data/master_ceramica_old_format",
                        help="Base directory of the old-format dataset")
    parser.add_argument("--out-dir", default="data/master_ceramica",
                        help="Base directory of the new-format dataset")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for equivalent edges",
    )
    parser.add_argument(
        "--max-sim-neighbors",
        type=int,
        default=100,
        help="Max equivalent neighbours per entity",
    )
    parser.add_argument(
        "--no-synonymy",
        action="store_true",
        help="Skip equivalent edge generation (fast mode)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output files"
    )
    args = parser.parse_args()

    old_stage1_dir = os.path.join(args.old_dir, "processed", "stage1")
    old_documents_path = os.path.join(args.old_dir, "raw", "dataset_corpus.json")
    out_stage1_dir = os.path.join(args.out_dir, "processed", "stage1")
    out_raw_dir = os.path.join(args.out_dir, "raw")

    # Check output dir
    out_nodes = os.path.join(out_stage1_dir, "nodes.csv")
    out_edges = os.path.join(out_stage1_dir, "edges.csv")
    out_relations = os.path.join(out_stage1_dir, "relations.csv")
    if not args.force and any(
        os.path.exists(p) for p in [out_nodes, out_edges, out_relations]
    ):
        logger.error("Output files already exist. Use --force to overwrite.")
        sys.exit(1)
    os.makedirs(out_stage1_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    logger.info("Loading inputs …")

    with open(old_documents_path, encoding="utf-8") as f:
        documents: dict[str, str] = json.load(f)
    logger.info("  dataset_corpus.json: %d entries", len(documents))

    doc2ent_path = os.path.join(old_stage1_dir, "document2entities.json")
    with open(doc2ent_path, encoding="utf-8") as f:
        document2entities: dict[str, list[str]] = json.load(f)
    logger.info("  document2entities.json: %d docs", len(document2entities))

    kg_path = os.path.join(old_stage1_dir, "kg.txt")
    kg_triples: list[tuple[str, str, str]] = []
    skipped = 0
    with open(kg_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(", ", 2)
            if len(parts) != 3:
                skipped += 1
                continue
            subj, rel, obj = parts
            if not subj or not rel or not obj:
                skipped += 1
                continue
            kg_triples.append((subj, rel, obj))
    logger.info("  kg.txt: %d triples loaded, %d skipped", len(kg_triples), skipped)

    # ------------------------------------------------------------------
    # 2. Parse markdown sections and build section edges
    # ------------------------------------------------------------------
    logger.info("Parsing markdown sections …")
    section_edges: list[
        tuple[str, str, str]
    ] = []  # (source, relation, target) — already normalized
    section_entity_names: set[str] = set()

    for doc_filename, content in documents.items():
        sections = parse_sections(content)
        if not sections:
            continue

        # prev_at_level maps heading level → (original_title, normalized_title)
        prev_at_level: dict[int, tuple[str, str]] = {}
        first_h2 = True

        for level, title in sections:
            norm_title = processing_phrases(title)
            if not norm_title:
                continue
            section_entity_names.add(norm_title)

            if level == 2:
                rel = "has section" if first_h2 else "contains section"
                section_edges.append((doc_filename, rel, norm_title))
                first_h2 = False
                prev_at_level = {2: (title, norm_title)}
            else:
                # Find nearest ancestor
                parent_norm = None
                for parent_level in range(level - 1, 1, -1):
                    if parent_level in prev_at_level:
                        parent_norm = prev_at_level[parent_level][1]
                        break
                if parent_norm is None:
                    # Fallback: attach to document
                    section_edges.append((doc_filename, "contains section", norm_title))
                else:
                    section_edges.append((parent_norm, "contains section", norm_title))
                prev_at_level[level] = (title, norm_title)

    logger.info("  Section edges: %d", len(section_edges))
    logger.info("  Section entity names: %d", len(section_entity_names))

    # ------------------------------------------------------------------
    # 3. Build entity set (normalized names)
    # ------------------------------------------------------------------
    entity_set: set[str] = set()

    for subj, _rel, obj in kg_triples:
        ns, no = processing_phrases(subj), processing_phrases(obj)
        if ns:
            entity_set.add(ns)
        if no:
            entity_set.add(no)

    for entities in document2entities.values():
        for ent in entities:
            ne = processing_phrases(ent)
            if ne:
                entity_set.add(ne)

    entity_set.update(section_entity_names)
    entity_set.discard("")
    logger.info("Total unique entity names: %d", len(entity_set))

    # ------------------------------------------------------------------
    # 4. Build nodes
    # ------------------------------------------------------------------
    logger.info("Building nodes …")
    nodes: list[dict] = []
    seen_names: set[str] = set()

    # Document nodes (original filename, not normalized)
    for doc_filename, content in documents.items():
        if doc_filename in seen_names:
            continue
        seen_names.add(doc_filename)
        nodes.append(
            {
                "name": doc_filename,
                "type": "document",
                "attributes": {"content": content},
            }
        )

    # Entity nodes
    for ent_name in sorted(entity_set):
        if ent_name in seen_names:
            continue
        seen_names.add(ent_name)
        nodes.append(
            {
                "name": ent_name,
                "type": "entity",
                "attributes": {},
            }
        )

    logger.info(
        "  Nodes: %d (%d document, %d entity)",
        len(nodes),
        len(documents),
        len(entity_set),
    )

    # ------------------------------------------------------------------
    # 5. Build edges (semantic + is_mentioned_in + sections)
    # ------------------------------------------------------------------
    logger.info("Building edges …")
    # Use set for deduplication: (source, relation, target)
    edge_set: set[tuple[str, str, str]] = set()
    ordered_edges: list[tuple[str, str, str]] = []

    def add_edge(src: str, rel: str, tgt: str) -> None:
        if not src or not tgt:
            return
        key = (src, rel, tgt)
        if key not in edge_set:
            edge_set.add(key)
            ordered_edges.append(key)

    # Semantic edges from kg.txt
    for subj, rel, obj in kg_triples:
        add_edge(processing_phrases(subj), rel, processing_phrases(obj))

    # is_mentioned_in edges (entity → document)
    node_name_set = seen_names  # all valid node names
    for doc_filename, entities in document2entities.items():
        if doc_filename not in node_name_set:
            continue
        for ent in entities:
            ne = processing_phrases(ent)
            if ne and ne in entity_set:
                add_edge(ne, "is_mentioned_in", doc_filename)

    # Section edges
    for src, rel, tgt in section_edges:
        add_edge(src, rel, tgt)

    logger.info("  Edges before synonymy: %d", len(ordered_edges))

    # ------------------------------------------------------------------
    # 6. Synonymy edges via vLLM embeddings (optional)
    # ------------------------------------------------------------------
    if not args.no_synonymy:
        embed_base_url = os.environ.get(
            "VLLM_EMBED_BASE_URL", "http://localhost:8083/v1"
        )
        embed_model = os.environ.get("VLLM_EMBED_MODEL")
        embed_api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

        if not embed_model:
            logger.warning(
                "VLLM_EMBED_MODEL not set — skipping synonymy edges. "
                "Set it in .env or use --no-synonymy."
            )
        else:
            logger.info(
                "Generating synonymy edges via vLLM (%s @ %s) …",
                embed_model,
                embed_base_url,
            )
            try:
                from gfmrag.graph_index_construction.entity_linking_model import (
                    VLLMELModel,
                )

                el_model = VLLMELModel(
                    model_name=embed_model,
                    api_base=embed_base_url,
                    api_key=embed_api_key,
                    root="tmp/migration_cache",
                )
                unique_phrases = sorted(entity_set)
                logger.info("  Indexing %d entity phrases …", len(unique_phrases))
                el_model.index(unique_phrases)

                logger.info("  Computing top-%d neighbours …", args.max_sim_neighbors)
                sim_neighbors = el_model(unique_phrases, topk=args.max_sim_neighbors)

                equiv_count = 0
                for phrase, neighbors in sim_neighbors.items():
                    if len(re.sub(r"[^A-Za-z0-9]", "", phrase)) <= 2:
                        continue
                    num_nns = 0
                    for n in neighbors:
                        if (
                            n["norm_score"] < args.threshold
                            or num_nns >= args.max_sim_neighbors
                        ):
                            break
                        if n["entity"] != phrase:
                            add_edge(phrase, "equivalent", n["entity"])
                            equiv_count += 1
                            num_nns += 1

                logger.info("  Synonymy edges added: %d", equiv_count)
            except RuntimeError as e:
                logger.error(
                    "vLLM server unavailable: %s — skipping synonymy edges.", e
                )
            except Exception as e:
                logger.error("Error generating synonymy edges: %s", e)
                raise
    else:
        logger.info("Skipping synonymy edges (--no-synonymy)")

    logger.info("Total edges: %d", len(ordered_edges))

    # ------------------------------------------------------------------
    # 7. Build relations
    # ------------------------------------------------------------------
    relations_set: set[str] = set()
    for _src, rel, _tgt in ordered_edges:
        relations_set.add(rel)

    # ------------------------------------------------------------------
    # 8. Write CSVs
    # ------------------------------------------------------------------
    logger.info("Writing CSVs to %s …", out_stage1_dir)

    # nodes.csv — attributes column as str() of dict
    nodes_rows = [
        {"name": n["name"], "type": n["type"], "attributes": str(n["attributes"])}
        for n in nodes
    ]
    nodes_df = pd.DataFrame(nodes_rows)
    nodes_df.to_csv(out_nodes, index=False)
    logger.info("  nodes.csv: %d rows", len(nodes_df))

    # edges.csv
    edges_rows = [
        {"source": src, "relation": rel, "target": tgt, "attributes": "{}"}
        for src, rel, tgt in ordered_edges
    ]
    edges_df = pd.DataFrame(edges_rows)
    edges_df.to_csv(out_edges, index=False)
    logger.info("  edges.csv: %d rows", len(edges_df))

    # relations.csv
    relations_rows = [
        {"name": rel, "attributes": "{}"} for rel in sorted(relations_set)
    ]
    relations_df = pd.DataFrame(relations_rows)
    relations_df.to_csv(out_relations, index=False)
    logger.info("  relations.csv: %d rows", len(relations_df))

    # ------------------------------------------------------------------
    # 9. Copy raw folder, renaming dataset_corpus.json -> documents.json
    #    and normalising QA field names in train/test JSON files.
    # ------------------------------------------------------------------
    out_documents = os.path.join(out_raw_dir, "documents.json")
    if not args.force and os.path.exists(out_documents):
        logger.warning(
            "%s already exists — skipping raw copy. Use --force to overwrite.", out_documents
        )
    else:
        old_raw_dir = os.path.join(args.old_dir, "raw")
        os.makedirs(out_raw_dir, exist_ok=True)
        for fname in os.listdir(old_raw_dir):
            src = os.path.join(old_raw_dir, fname)
            dst_name = "documents.json" if fname == "dataset_corpus.json" else fname
            dst = os.path.join(out_raw_dir, dst_name)

            if fname in ("train.json", "test.json"):
                with open(src, encoding="utf-8") as f:
                    qa_data = json.load(f)
                renamed = 0
                for sample in qa_data:
                    if "supporting_facts" in sample and "supporting_documents" not in sample:
                        sample["supporting_documents"] = sample.pop("supporting_facts")
                        renamed += 1
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(qa_data, f, ensure_ascii=False, indent=4)
                logger.info("  raw: %s -> %s  (renamed supporting_facts in %d samples)", fname, dst_name, renamed)

                # If samples have pre-annotated entities, write processed/stage1/ directly.
                has_entities = qa_data and (
                    "question_entities" in qa_data[0] and "supporting_entities" in qa_data[0]
                )
                processed_out = os.path.join(out_stage1_dir, fname)
                if has_entities and (args.force or not os.path.exists(processed_out)):
                    processed_data = []
                    for sample in qa_data:
                        target_docs = sample.get("supporting_documents", sample.get("supporting_facts", []))
                        processed_data.append({
                            **sample,
                            "supporting_documents": target_docs,
                            "start_type": ["entity"],
                            "target_type": ["entity", "document"],
                            "start_nodes": {
                                "entity": [processing_phrases(e) for e in sample.get("question_entities", [])],
                            },
                            "target_nodes": {
                                "entity": [processing_phrases(e) for e in sample.get("supporting_entities", [])],
                                "document": target_docs,
                            },
                        })
                    with open(processed_out, "w", encoding="utf-8") as f:
                        json.dump(processed_data, f, ensure_ascii=False, indent=4)
                    logger.info("  stage1: %s written (%d samples, fast migration)", fname, len(processed_data))
            else:
                shutil.copy2(src, dst)
                logger.info("  raw: %s -> %s", fname, dst_name)

    logger.info("Migration complete.")


if __name__ == "__main__":
    main()
