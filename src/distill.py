"""Distills fitted Bradley-Terry ratings into a pointwise cross-encoder reranker.

Two backends behind a uniform `Reranker.predict(pairs) -> np.ndarray` interface:
  - `TinyOfflineReranker`: a small from-scratch torch MLP over cheap lexical features
    (BM25 score, token overlap, length ratio). No network access, no HF download --
    used for --smoke so the pipeline is fully testable offline.
  - `CrossEncoderReranker`: wraps `sentence_transformers.CrossEncoder` (default base
    `cross-encoder/ms-marco-MiniLM-L-6-v2`, per plan.md), trained via regression on the
    normalized rating labels. Used for full (non-smoke) runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TrainingExample:
    query_id: str
    query_text: str
    doc_id: str
    doc_text: str
    label: float  # normalized relevance target in [0, 1], derived from fitted ratings


def ratings_to_labels(ratings: np.ndarray) -> np.ndarray:
    """Per-query min-max normalization of ratings into [0, 1] regression targets.
    (A sigmoid squash would saturate for the large-margin skeleton battles; min-max keeps
    the full within-query rank spread usable as a training signal.)"""
    lo, hi = ratings.min(), ratings.max()
    if hi - lo < 1e-9:
        return np.full_like(ratings, 0.5)
    return (ratings - lo) / (hi - lo)


def build_training_examples(
    pools_by_query: dict[str, "object"],  # query_id -> CandidatePool (from beir_loader)
    ratings_per_query: dict[str, np.ndarray],
    doc_index_per_query: dict[str, dict[str, int]],
) -> list[TrainingExample]:
    examples = []
    for qid, ratings in ratings_per_query.items():
        pool = pools_by_query[qid]
        labels = ratings_to_labels(ratings)
        doc_index = doc_index_per_query[qid]
        for doc_id in doc_index:
            idx = doc_index[doc_id]
            doc_text = pool.doc_texts[pool.doc_ids.index(doc_id)]
            examples.append(TrainingExample(qid, pool.query_text, doc_id, doc_text, float(labels[idx])))
    return examples


class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...


def _lexical_features(query: str, doc: str) -> list[float]:
    q_tokens = set(query.lower().split())
    d_tokens = set(doc.lower().split())
    overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
    len_ratio = len(d_tokens) / max(len(q_tokens), 1)
    jaccard = len(q_tokens & d_tokens) / max(len(q_tokens | d_tokens), 1)
    return [overlap, min(len_ratio, 10.0) / 10.0, jaccard]


class _TinyMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TinyOfflineReranker:
    """Offline stand-in "cross-encoder" for --smoke: a tiny MLP over lexical features,
    trained from scratch, no network/HF download. Not a real cross-encoder -- exists only
    so Phase 2's pipeline (curriculum -> distill -> evaluate) is fully testable offline."""

    def __init__(self, seed: int = 0, epochs: int = 100, lr: float = 0.05):
        torch.manual_seed(seed)
        self.model = _TinyMLP(n_features=3)
        self.epochs = epochs
        self.lr = lr

    def fit(self, examples: list[TrainingExample]) -> None:
        X = torch.tensor(
            [_lexical_features(e.query_text, e.doc_text) for e in examples], dtype=torch.float32
        )
        y = torch.tensor([e.label for e in examples], dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        X = torch.tensor([_lexical_features(q, d) for q, d in pairs], dtype=torch.float32)
        with torch.no_grad():
            return self.model(X).numpy()


class CrossEncoderReranker:
    """Wraps sentence-transformers CrossEncoder for full (non-smoke) runs."""

    def __init__(self, base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", seed: int = 0):
        from sentence_transformers import CrossEncoder

        torch.manual_seed(seed)
        self.model = CrossEncoder(base_model, num_labels=1)

    def fit(self, examples: list[TrainingExample], epochs: int = 1, batch_size: int = 16) -> None:
        from sentence_transformers import InputExample
        from torch.utils.data import DataLoader

        train_samples = [
            InputExample(texts=[e.query_text, e.doc_text], label=e.label) for e in examples
        ]
        loader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
        self.model.fit(train_dataloader=loader, epochs=epochs, show_progress_bar=False)

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return self.model.predict(list(pairs))


def train_reranker(
    examples: list[TrainingExample], smoke: bool, seed: int = 0, base_model: str | None = None
) -> Reranker:
    if smoke:
        reranker = TinyOfflineReranker(seed=seed)
    else:
        reranker = CrossEncoderReranker(
            base_model=base_model or "cross-encoder/ms-marco-MiniLM-L-6-v2", seed=seed
        )
    reranker.fit(examples)
    return reranker
