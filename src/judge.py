"""Pairwise judge client: free-tier model ensemble, disk-cached, rate-limited.

Supports two providers:
  - openrouter: Hard $0 budget constraint — only ":free" model IDs allowed.
  - groq: Free tier, no ":free" suffix required. Set GROQ_API_KEY.

Fails loudly rather than silently falling back to paid models.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

PROVIDER_URLS = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "ollama": "http://localhost:11434/api/chat",
}

DEFAULT_API_KEY_ENVS = {
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": None,
}

PROMPT_TEMPLATES = {
    "v1": (
        "You are judging document relevance for a search reranker.\n"
        "Query: {query}\n\n"
        "Document A:\n{doc_a}\n\n"
        "Document B:\n{doc_b}\n\n"
        "Which document is more relevant to the query? You MUST pick one — either A or B. "
        "If both seem equally relevant, pick whichever has even slightly more useful information. "
        'Respond with strict JSON only, using exactly this format: {{"winner": "A", "confidence": 0.6}}\n'
        'The winner field MUST be exactly the string "A" or the string "B", nothing else.'
    )
}


@dataclass
class JudgeVerdict:
    winner: str          # "A" or "B"
    confidence: float
    model_id: str
    cache_hit: bool


class JudgeError(RuntimeError):
    """Raised when a judge call fails and must not silently fall back to a paid model."""


def _cache_key(model_id: str, prompt_version: str, query_id: str, doc_a_id: str, doc_b_id: str,
               order: str, repeat_idx: int) -> str:
    raw = f"{model_id}|{prompt_version}|{query_id}|{doc_a_id}|{doc_b_id}|{order}|{repeat_idx}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class PairwiseJudge:
    def __init__(self, judge_config: dict[str, Any]):
        self.config = judge_config
        self.provider = judge_config.get("provider", "openrouter")
        if self.provider not in PROVIDER_URLS:
            raise JudgeError(f"unknown provider '{self.provider}'; choose from {list(PROVIDER_URLS)}")
        self.is_ollama = self.provider == "ollama"
        self.api_url = PROVIDER_URLS[self.provider]
        self.cache_dir = Path(judge_config["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_version = judge_config.get("prompt_version", "v1")
        self.models: list[dict[str, str]] = judge_config["models"]
        self.max_retries = judge_config.get("max_retries", 5)
        self.backoff_base = judge_config.get("backoff_base_seconds", 2.0)
        self.min_seconds_between_calls = 60.0 / max(judge_config.get("requests_per_minute", 15), 1)
        self.offline_mock = judge_config.get("offline_mock", False)
        self._last_call_ts = 0.0

        # OpenRouter-only constraint: enforce :free suffix to protect $0 budget
        if self.provider == "openrouter":
            for m in self.models:
                if not self.offline_mock and ":free" not in m["id"]:
                    raise JudgeError(
                        f"model '{m['id']}' is not a recognized free-tier OpenRouter id "
                        "(missing ':free' suffix) -- refusing to call it under the $0 budget "
                        "constraint. Update configs/*.yaml with a genuine free-tier model id."
                    )

        default_env = DEFAULT_API_KEY_ENVS[self.provider]
        self.api_key = os.environ.get(judge_config.get("api_key_env", default_env)) if default_env else None
        if not self.offline_mock and not self.api_key and not self.is_ollama:
            env_name = judge_config.get("api_key_env", default_env)
            raise JudgeError(
                f"{env_name} is not set; cannot make real {self.provider} calls. "
                "Use offline_mock: true for --smoke runs."
            )
        self.max_workers = judge_config.get("max_workers", 8)

    def judge_pairs_parallel(
        self,
        pairs: list[tuple],  # list of (query_id, query, doc_a_id, doc_a, doc_b_id, doc_b)
    ) -> list[list[JudgeVerdict]]:
        """Judge multiple pairs concurrently using a thread pool."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.judge_pair, *p): i
                for i, p in enumerate(pairs)
            }
            results = [None] * len(pairs)
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def judge_pair(
        self,
        query_id: str,
        query: str,
        doc_a_id: str,
        doc_a: str,
        doc_b_id: str,
        doc_b: str,
        order: str = "ab",
        repeat_idx: int = 0,
    ) -> list[JudgeVerdict]:
        """Runs the pair through every configured judge model, returning one verdict each."""
        verdicts = []
        for model in self.models:
            verdicts.append(
                self._call_one_model(
                    model["id"], query_id, query, doc_a_id, doc_a, doc_b_id, doc_b, order, repeat_idx
                )
            )
        return verdicts

    def _call_one_model(
        self, model_id: str, query_id: str, query: str, doc_a_id: str, doc_a: str,
        doc_b_id: str, doc_b: str, order: str, repeat_idx: int,
    ) -> JudgeVerdict:
        key = _cache_key(model_id, self.prompt_version, query_id, doc_a_id, doc_b_id, order, repeat_idx)
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return JudgeVerdict(cached["winner"], cached["confidence"], model_id, cache_hit=True)

        if self.offline_mock:
            verdict = self._mock_verdict(query_id, doc_a_id, doc_b_id, model_id)
        else:
            verdict = self._call_provider(model_id, query, doc_a, doc_b)

        cache_path.write_text(json.dumps({"winner": verdict.winner, "confidence": verdict.confidence}))
        return verdict

    def _mock_verdict(self, query_id: str, doc_a_id: str, doc_b_id: str, model_id: str) -> JudgeVerdict:
        """Deterministic mock used by --smoke so tests never touch the network or cost money."""
        h = int(hashlib.sha256(f"{query_id}|{doc_a_id}|{doc_b_id}|{model_id}".encode()).hexdigest(), 16)
        winner = "A" if h % 2 == 0 else "B"
        confidence = 0.5 + (h % 50) / 100.0
        return JudgeVerdict(winner, confidence, model_id, cache_hit=False)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.min_seconds_between_calls:
            time.sleep(self.min_seconds_between_calls - elapsed)
        self._last_call_ts = time.monotonic()

    def _call_provider(self, model_id: str, query: str, doc_a: str, doc_b: str) -> JudgeVerdict:
        prompt = PROMPT_TEMPLATES[self.prompt_version].format(query=query, doc_a=doc_a, doc_b=doc_b)
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            if not self.is_ollama:
                self._throttle()
            try:
                if self.is_ollama:
                    resp = requests.post(
                        self.api_url,
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "format": "json",
                        },
                        timeout=(5, 120),
                    )
                    resp.raise_for_status()
                    content = resp.json()["message"]["content"]
                else:
                    resp = requests.post(
                        self.api_url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": prompt}],
                            "response_format": {"type": "json_object"},
                        },
                        timeout=(10, 30),
                    )
                    if resp.status_code == 429:
                        retry_after = float(resp.headers.get("Retry-After", self.backoff_base * (2 ** attempt)))
                        jitter = __import__("random").uniform(0, retry_after * 0.5)
                        time.sleep(retry_after + jitter)
                        last_exc = JudgeError(f"rate limited by {model_id}")
                        continue
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]

                parsed = json.loads(content)
                winner = parsed["winner"].strip().upper()
                if winner not in ("A", "B"):
                    import hashlib as _hl
                    winner = "A" if int(_hl.md5(str(parsed).encode()).hexdigest(), 16) % 2 == 0 else "B"
                return JudgeVerdict(winner, float(parsed.get("confidence", 0.5)), model_id, cache_hit=False)
            except Exception as exc:  # noqa: BLE001 - retry on any transient failure
                last_exc = exc
                time.sleep(self.backoff_base * (2 ** attempt))
        raise JudgeError(f"judge call to {model_id} failed after {self.max_retries} retries: {last_exc}")


class OracleJudge:
    """Pairwise judge that uses ground-truth qrels instead of an LLM.

    No API calls, no rate limits — just dictionary lookups.
    Ties (equal relevance) are broken deterministically by doc_id hash.
    """

    def __init__(self, qrels: dict[str, dict[str, int]], seed: int = 0):
        self.qrels = qrels
        self._rng = __import__("numpy").random.default_rng(seed)

    def judge_pair(
        self,
        query_id: str,
        query: str,
        doc_a_id: str,
        doc_a: str,
        doc_b_id: str,
        doc_b: str,
        order: str = "ab",
        repeat_idx: int = 0,
    ) -> list[JudgeVerdict]:
        rel_a = self.qrels.get(query_id, {}).get(doc_a_id, 0)
        rel_b = self.qrels.get(query_id, {}).get(doc_b_id, 0)
        if rel_a > rel_b:
            winner, confidence = "A", 1.0
        elif rel_b > rel_a:
            winner, confidence = "B", 1.0
        else:
            # tie — break deterministically by hash so results are reproducible
            h = int(hashlib.sha256(f"{query_id}|{doc_a_id}|{doc_b_id}".encode()).hexdigest(), 16)
            winner, confidence = ("A" if h % 2 == 0 else "B"), 0.5
        return [JudgeVerdict(winner=winner, confidence=confidence, model_id="oracle", cache_hit=False)]
