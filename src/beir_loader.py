"""BEIR dataset loading (FiQA, SciFact) and top-k candidate pool construction.

Interface is dataset-agnostic (works on any BEIR-style corpus/queries/qrels triple) so a
future code-retrieval BEIR set drops in without touching downstream code, per plan.md.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi

BEIR_DATASETS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


@dataclass
class CandidatePool:
    query_id: str
    query_text: str
    doc_ids: list[str]           # length == pool_size (or fewer if corpus is tiny)
    doc_texts: list[str]
    relevant_doc_ids: set[str]   # ground-truth qrels for this query, for eval only
    bm25_scores: dict[str, float]  # doc_id -> BM25 score for this query, cheap pre-fit difficulty signal


def download_beir_dataset(name: str, data_dir: str) -> str:
    """Downloads+unzips a BEIR dataset if not already present; returns the dataset folder path."""
    url = BEIR_DATASETS_URL.format(name)
    out_dir = beir_util.download_and_unzip(url, data_dir)
    return out_dir


def load_beir(name: str, data_dir: str, split: str = "test"):
    dataset_dir = download_beir_dataset(name, data_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_dir).load(split=split)
    return corpus, queries, qrels


def _doc_text(doc: dict) -> str:
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    return f"{title}\n{text}".strip()


def build_candidate_pools(
    corpus: dict, queries: dict, qrels: dict, pool_size: int, seed: int = 0,
    max_queries: int | None = None,
) -> list[CandidatePool]:
    """For each query, builds a pool of `pool_size` candidate docs via BM25 top-k, guaranteeing
    at least one known-relevant doc (from qrels) is included when available so near-tie and
    clear-win pairs both actually occur within the pool."""
    rng = random.Random(seed)
    doc_ids = list(corpus.keys())
    tokenized_corpus = [_doc_text(corpus[d]).lower().split() for d in doc_ids]
    bm25 = BM25Okapi(tokenized_corpus)

    query_ids = list(queries.keys())
    rng.shuffle(query_ids)
    if max_queries is not None:
        query_ids = query_ids[:max_queries]

    pools = []
    for qid in query_ids:
        query_text = queries[qid]
        tokenized_query = query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(doc_ids)), key=lambda i: scores[i], reverse=True)

        relevant = {d for d, rel in qrels.get(qid, {}).items() if rel > 0}
        pool_ids: list[str] = []
        # guarantee at least one relevant doc in the pool if one exists in the corpus
        for rel_id in sorted(relevant):
            if rel_id in corpus and len(pool_ids) < pool_size:
                pool_ids.append(rel_id)
        for i in ranked:
            if len(pool_ids) >= pool_size:
                break
            d = doc_ids[i]
            if d not in pool_ids:
                pool_ids.append(d)

        rng.shuffle(pool_ids)  # randomize pool order so fallback isn't BM25 rank
        pool_bm25 = {doc_ids[i]: float(scores[i]) for i in ranked if doc_ids[i] in pool_ids}
        pools.append(
            CandidatePool(
                query_id=qid,
                query_text=query_text,
                doc_ids=pool_ids,
                doc_texts=[_doc_text(corpus[d]) for d in pool_ids],
                relevant_doc_ids=relevant,
                bm25_scores=pool_bm25,
            )
        )
    return pools


def generate_synthetic_dataset(n_queries: int, pool_size: int, seed: int = 0):
    """Small self-contained synthetic corpus/queries/qrels for --smoke runs: no network access,
    no BEIR download, deterministic given `seed`. Text is toy but non-degenerate so BM25 and the
    mock judge produce varied, reproducible orderings."""
    rng = random.Random(seed)
    topics = ["loans", "stocks", "bonds", "taxes", "insurance", "budgeting", "credit", "retirement"]
    corpus, queries, qrels = {}, {}, {}
    n_docs_per_query = max(pool_size * 2, 10)
    doc_counter = 0
    for qi in range(n_queries):
        topic = topics[qi % len(topics)]
        qid = f"q{qi}"
        queries[qid] = f"What are the best practices for {topic} management?"
        qrels[qid] = {}
        for _ in range(n_docs_per_query):
            did = f"d{doc_counter}"
            doc_counter += 1
            is_relevant = rng.random() < 0.3
            filler = rng.choice(topics)
            text = (
                f"Guide to {topic} strategies and {filler} considerations."
                if is_relevant
                else f"An unrelated article about {filler} that barely mentions {topic}."
            )
            corpus[did] = {"title": f"Doc {did}", "text": text}
            if is_relevant:
                qrels[qid][did] = 1
    return corpus, queries, qrels
