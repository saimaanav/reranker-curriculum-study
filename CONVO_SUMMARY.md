# zelo-curriculum-study — Conversation Summary

**Goal:** Portfolio project for ZeroEntropy. Test whether difficulty-scheduled pairwise comparison curricula improve reranker training, inspired by their zELO method. $0 budget — free/local tools only.

---

## Architecture (final)
- **Judge:** Ollama (llama3 8B, local, M5 Mac, no rate limits, ~0.5-1s/call)
- **Reranker:** BTRatingReranker — fits Bradley-Terry ratings from judged pairs, uses ratings directly as relevance scores. No neural training.
- **4 arms:** random, difficulty_curriculum, anti_curriculum, compositional_curriculum
- **Datasets downloaded:** FiQA (`data/raw/fiqa`), NFCorpus (`data/raw/nfcorpus`), TREC-COVID (`data/raw/trec-covid/trec-covid`)
- **Key files:** `src/judge.py`, `scripts/run_phase2.py`, `src/beir_loader.py`, `configs/fiqa.yaml`

---

## Problem Log

### P1 — OpenRouter model 404s
Free-tier models (`meta-llama/llama-3.1-8b-instruct:free`, `google/gemini-flash-1.5-8b:free`) were silently removed from OpenRouter. Queried `/api/v1/models` to find still-available ones. Eventually abandoned OpenRouter entirely.

### P2 — Groq rate limits
Groq advertises 30 RPM but actual sustained throughput was ~1-3 calls/min. ThreadPoolExecutor parallelism made it worse — all threads 429'd simultaneously then backed off together. Added retry+jitter logic to `judge.py`. Still too slow for a full run. Eventually switched away from Groq.

### P3 — MiniLM CrossEncoder training too slow
Original design trained a CrossEncoder on each arm's judged pairs. On CPU: 15-30 min per training run × 4 arms × 4 checkpoints = hours per experiment. Scrapped entirely. Replaced with BTRatingReranker — no neural training needed.

### P4 — Calibration call volume
Eval calibration was building 40 pairs × 194 queries = ~7,760 API calls just for calibration. Cut to 5 pairs/query and capped `max_queries=20` for validation runs.

### P5 — Ollama API format wrong
Added Ollama provider to `judge.py` but initially used OpenAI response format. Fix: use `POST /api/chat`, set `stream: false` and `format: json`, read `response["message"]["content"]` not `response["choices"][0]["message"]["content"]`.

### P6 — NFCorpus: 78% ties → null results (first null)
All 4 arms produced identical NDCG/Recall/MRR at every checkpoint (±0.000 std). Root cause: 78.4% of all pairs in NFCorpus k=20 pools are ties (both docs have equal relevance score). B-T MLE saturates immediately — it has no signal to differentiate rankings regardless of which pairs are scheduled first.

### P7 — TREC-COVID: all eval pools 100% relevant
Switched to TREC-COVID which had 43.3% informative pairs (ideal ratio). Still got NDCG=1.0 for all arms. Root cause: TREC-COVID qrels label almost every doc in a k=20 pool as relevant (20/20 relevant per eval query). Any ranking trivially achieves NDCG=1.0. Dataset not discriminative enough at k=20.

### P8 — BM25 fallback masking null results
Unseen docs score 0.0 from BTRatingReranker. `np.argsort(-scores)` on all-zeros preserves original pool order. Pool order was: relevant docs first, then BM25 top docs — so even with zero B-T information, every pool was already perfectly ranked. Fixed by shuffling `pool_ids` in `beir_loader.py` after construction so fallback order is random.

### P9 — Core architecture bug: evaluating on never-compared docs (THE real fix)
Even after P8 fix, all arms still identical (±0.000). Root cause found: BTRatingReranker only stores scores for docs it has *compared*. The code called `evaluate_ranking_on_pools(reranker, eval_pools)` — but eval pool docs were never part of any arm's training comparisons. They all score 0.0. Same shuffled-random ranking for every arm, every seed → identical NDCG.

**Fix applied (`scripts/run_phase2.py` line 229):**
```python
# Before (broken):
ranking_metrics = evaluate_ranking_on_pools(reranker, eval_pools)

# After (fixed):
ranking_metrics = evaluate_ranking_on_pools(reranker, train_pools)
```
B-T is transductive — it only knows docs it has seen in comparisons. Evaluating on train_pools measures exactly what we want: did this arm's pair selection produce better B-T ratings?

---

## Current State

- All 9 fixes applied
- FiQA config: Ollama, `max_queries=20`, `budget=100`, `seeds=[0,1,2]`, 4 arms = 12 runs
- FiQA has ~2.2 relevant docs per eval pool on average — NDCG is actually meaningful
- **P9 fix not yet validated** — phase 2 has not been re-run since this fix

## Next Step
```bash
cd ~/zelo-curriculum-study
source .venv/bin/activate
python3 scripts/run_phase2.py --config configs/fiqa.yaml
```
Estimated runtime: ~10-15 min. This should produce non-null, differentiated results across arms for the first time.
