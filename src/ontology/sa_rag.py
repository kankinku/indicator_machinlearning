from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import config
from src.shared.logger import get_logger

logger = get_logger("ontology.sa_rag")


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Node:
    node_id: str
    node_type: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ActivationResult:
    query: str
    seed_nodes: List[str]
    activated_nodes: Dict[str, float]
    selected_chunks: List[Node]
    relation_texts: List[str]
    context: str
    debug: Dict[str, object] = field(default_factory=dict)


class KnowledgeGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.adjacency: Dict[str, List[Edge]] = {}

    def add_node(self, node: Node) -> None:
        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node
            self.adjacency[node.node_id] = []

    def add_edge(self, src: str, dst: str, text: str, bidirectional: bool = True) -> None:
        edge = Edge(src=src, dst=dst, text=text)
        self.adjacency.setdefault(src, []).append(edge)
        if bidirectional:
            reverse_edge = Edge(src=dst, dst=src, text=text)
            self.adjacency.setdefault(dst, []).append(reverse_edge)

    def neighbors(self, node_id: str) -> List[Edge]:
        return self.adjacency.get(node_id, [])


class SpreadingActivationRAG:
    def __init__(self):
        self.graph = KnowledgeGraph()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.node_vectors: Dict[str, np.ndarray] = {}
        self.edge_vectors: Dict[Tuple[str, str, str], np.ndarray] = {}
        self.entity_desc_nodes: List[str] = []

    def index_documents(self, documents: Iterable[Document]) -> None:
        try:
            chunks = self._chunk_documents(documents)
            self._build_graph(chunks)
            self._build_embeddings()
        except Exception as exc:
            self._record_error("index_documents", exc)
            raise

    def query(self, query_text: str) -> ActivationResult:
        if not self.vectorizer:
            raise RuntimeError("Index is empty. Run index_documents first.")

        try:
            q_vec = self._embed_text(query_text)
            seeds = self._select_seeds(q_vec)
            sub_nodes = self._subgraph_nodes(seeds)
            activations = self._spreading_activation(q_vec, seeds, sub_nodes)
            selected_nodes = {nid: score for nid, score in activations.items() if score >= config.SA_THRESHOLD}
            chunks, rel_texts = self._collect_context(selected_nodes)
            context = self._format_context(chunks, rel_texts)
            return ActivationResult(
                query=query_text,
                seed_nodes=seeds,
                activated_nodes=selected_nodes,
                selected_chunks=chunks,
                relation_texts=rel_texts,
                context=context,
                debug={
                    "total_nodes": len(self.graph.nodes),
                    "subgraph_nodes": len(sub_nodes),
                    "activation_count": len(selected_nodes),
                },
            )
        except Exception as exc:
            self._record_error("query", exc)
            raise

    def _chunk_documents(self, documents: Iterable[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            words = doc.text.split()
            if not words:
                continue
            step = max(1, config.SA_CHUNK_SIZE_WORDS - config.SA_CHUNK_OVERLAP_WORDS)
            for i in range(0, len(words), step):
                chunk_words = words[i:i + config.SA_CHUNK_SIZE_WORDS]
                if not chunk_words:
                    continue
                chunk_text = " ".join(chunk_words)
                chunk_id = f"{doc.doc_id}_chunk_{len(chunks)}"
                chunks.append(Document(doc_id=chunk_id, text=chunk_text, metadata={"source": doc.doc_id}))
        return chunks

    def _build_graph(self, chunks: List[Document]) -> None:
        entity_map: Dict[str, str] = {}
        for chunk in chunks:
            chunk_node = Node(node_id=f"chunk:{chunk.doc_id}", node_type="chunk", text=chunk.text)
            self.graph.add_node(chunk_node)
            entities = self._extract_entities(chunk.text)
            entity_ids = []
            for entity in entities:
                if entity not in entity_map:
                    ent_node_id = f"entity:{entity}"
                    desc_node_id = f"entity_desc:{entity}"
                    entity_map[entity] = desc_node_id
                    self.graph.add_node(Node(node_id=ent_node_id, node_type="entity", text=entity))
                    desc_text = self._entity_description(entity, chunk.text)
                    self.graph.add_node(Node(node_id=desc_node_id, node_type="entity_desc", text=desc_text))
                    self.graph.add_edge(ent_node_id, desc_node_id, text=f"description of {entity}")
                entity_ids.append(entity_map[entity])
                self.graph.add_edge(chunk_node.node_id, entity_map[entity], text=f"mentions {entity}")

            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    e1 = entity_ids[i]
                    e2 = entity_ids[j]
                    self.graph.add_edge(e1, e2, text="co_occurs")

        self.entity_desc_nodes = [nid for nid, node in self.graph.nodes.items() if node.node_type == "entity_desc"]

    def _build_embeddings(self) -> None:
        node_texts = [node.text for node in self.graph.nodes.values()]
        edge_texts = [edge.text for edges in self.graph.adjacency.values() for edge in edges]
        corpus = node_texts + edge_texts
        if not corpus:
            raise ValueError("No content to embed.")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        matrix = self.vectorizer.fit_transform(corpus)
        vectors = matrix.toarray()

        node_count = len(node_texts)
        for idx, node_id in enumerate(self.graph.nodes.keys()):
            self.node_vectors[node_id] = vectors[idx]

        offset = node_count
        edge_items = []
        for src, edges in self.graph.adjacency.items():
            for edge in edges:
                edge_items.append((src, edge.dst, edge.text))
        for i, edge_key in enumerate(edge_items):
            self.edge_vectors[edge_key] = vectors[offset + i]

    def _embed_text(self, text: str) -> np.ndarray:
        vec = self.vectorizer.transform([text]).toarray()[0]
        return vec

    def _select_seeds(self, query_vec: np.ndarray) -> List[str]:
        scored = []
        for node_id in self.entity_desc_nodes:
            score = self._cosine(query_vec, self.node_vectors[node_id])
            scored.append((score, node_id))
        scored.sort(reverse=True, key=lambda x: x[0])
        top_k = max(1, min(config.SA_TOP_K, len(scored)))
        return [node_id for _, node_id in scored[:top_k]]

    def _subgraph_nodes(self, seeds: List[str]) -> List[str]:
        if not seeds:
            return []
        visited = set(seeds)
        frontier = list(seeds)
        for _ in range(config.SA_N_HOP):
            next_frontier = []
            for node_id in frontier:
                for edge in self.graph.neighbors(node_id):
                    if edge.dst not in visited:
                        visited.add(edge.dst)
                        next_frontier.append(edge.dst)
            frontier = next_frontier
            if not frontier:
                break
        return list(visited)

    def _spreading_activation(
        self,
        query_vec: np.ndarray,
        seeds: List[str],
        sub_nodes: List[str],
    ) -> Dict[str, float]:
        activations: Dict[str, float] = {node_id: 0.0 for node_id in sub_nodes}
        queue: List[str] = []
        for seed in seeds:
            if seed in activations:
                activations[seed] = 1.0
                queue.append(seed)

        in_queue = set(queue)
        while queue:
            current = queue.pop(0)
            in_queue.discard(current)
            for edge in self.graph.neighbors(current):
                if edge.dst not in activations:
                    continue
                w = self._edge_weight(query_vec, edge)
                if w <= 0:
                    continue
                updated = min(1.0, activations[edge.dst] + activations[current] * w)
                if updated > activations[edge.dst] + 1e-6:
                    activations[edge.dst] = updated
                    if edge.dst not in in_queue:
                        queue.append(edge.dst)
                        in_queue.add(edge.dst)
        return activations

    def _edge_weight(self, query_vec: np.ndarray, edge: Edge) -> float:
        key = (edge.src, edge.dst, edge.text)
        vec = self.edge_vectors.get(key)
        if vec is None:
            return 0.0
        raw = self._cosine(query_vec, vec)
        if raw <= config.SA_NORMALIZATION_C:
            return 0.0
        normalized = (raw - config.SA_NORMALIZATION_C) / (1.0 - config.SA_NORMALIZATION_C)
        return max(0.0, min(1.0, float(normalized)))

    def _collect_context(self, activated_nodes: Dict[str, float]) -> Tuple[List[Node], List[str]]:
        chunks: Dict[str, Node] = {}
        rel_texts: List[str] = []
        for node_id in activated_nodes:
            for edge in self.graph.neighbors(node_id):
                rel_texts.append(edge.text)
                if edge.dst.startswith("chunk:"):
                    chunks[edge.dst] = self.graph.nodes[edge.dst]
        rel_texts = sorted(set(rel_texts))
        return list(chunks.values()), rel_texts

    def _format_context(self, chunks: List[Node], relation_texts: List[str]) -> str:
        parts = []
        if relation_texts:
            parts.append("Relations:")
            parts.extend(f"- {text}" for text in relation_texts[:20])
        if chunks:
            parts.append("Documents:")
            for node in chunks[:config.SA_MAX_CONTEXT_CHUNKS]:
                parts.append(node.text)
        return "\n".join(parts)

    def _extract_entities(self, text: str) -> List[str]:
        stop_words = {"The", "This", "That", "With", "From", "Into", "And", "For", "A", "An"}
        candidates = re.findall(r"\b[A-Z][a-zA-Z0-9&-]+(?:\s+[A-Z][a-zA-Z0-9&-]+){0,2}\b", text)
        entities = []
        seen = set()
        for cand in candidates:
            if cand in stop_words:
                continue
            if cand.lower() in {"the", "and", "for"}:
                continue
            if cand not in seen:
                entities.append(cand)
                seen.add(cand)
        if not entities:
            tokens = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            freq: Dict[str, int] = {}
            for tok in tokens:
                freq[tok] = freq.get(tok, 0) + 1
            fallback = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
            entities = [tok.title() for tok, _ in fallback]
        return entities[:config.SA_MAX_ENTITIES_PER_CHUNK]

    def _entity_description(self, entity: str, text: str) -> str:
        sentences = re.split(r"[.!?]", text)
        for sent in sentences:
            if entity in sent:
                return f"{entity}: {sent.strip()}"
        return f"{entity}: {text[:200]}"

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def _record_error(self, stage: str, exc: Exception) -> None:
        try:
            import traceback
            base_dir = config.LOG_DIR / "errors"
            base_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "error_id": str(uuid4()),
                "timestamp": time.time(),
                "environment": getattr(config, "ENV", "default"),
                "stage": stage,
                "message": str(exc),
                "stack_trace": traceback.format_exc(),
                "resolution": "inspect_logs",
                "related_config": {
                    "SA_CHUNK_SIZE_WORDS": config.SA_CHUNK_SIZE_WORDS,
                    "SA_CHUNK_OVERLAP_WORDS": config.SA_CHUNK_OVERLAP_WORDS,
                    "SA_TOP_K": config.SA_TOP_K,
                    "SA_N_HOP": config.SA_N_HOP,
                    "SA_NORMALIZATION_C": config.SA_NORMALIZATION_C,
                    "SA_THRESHOLD": config.SA_THRESHOLD,
                },
            }
            error_path = base_dir / f"sa_rag_{payload['error_id']}.json"
            error_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            logger.error("Failed to persist SA-RAG error.", exc_info=True)
