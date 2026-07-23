# zelo-curriculum-study — Session Context

## Project Goal
Testing whether pairwise comparison scheduling order affects Bradley-Terry reranker quality on FiQA (BEIR benchmark), inspired by ZeroEntropy's zELO method.  Presenting at YC Startup School YCML Research Symposium, July 26 2026, Chase Center (~200 people, mingle-style).

## Experiment Design (3 arms)
- **random_cycles**: zELO-style, random cycles ensure every doc gets compared equally (principled baseline)
- **compositional**: easy pairs first (high difficulty score = easy), builds B-T skeleton before confusing near-ties
- **anti_compositional**: exact reverse, hard pairs first (true ablation)

**Difficulty score** (higher = easier):
```
difficulty = 0.7 * normalize(margin) + 0.3 * (1 - normalize(semantic_sim))
```
- `margin` = |BM25_a - BM25_b| — large gap = one doc clearly better = easy
- `semantic_sim` = sentence embedding cosine similarity — similar docs are confusingly close = hard
- Coefficients justified by margin sampling literature (margin dominates) + hard negative mining (semantic sim catches confusing pairs)

**Settings**: FiQA, 35 queries, k=20 pool, budget=300 judge calls, 5 seeds (0–4), Ollama Llama3 8B judge

## Old Results (4-arm design, stale — need re-run)
| Arm | NDCG@10 | vs. random |
|---|---|---|
| compositional_curriculum | 0.388 ± 0.020 | +14% |
| difficulty_curriculum | 0.369 ± 0.000 | +8% |
| random | 0.340 ± 0.032 | baseline |
| anti_curriculum | 0.221 ± 0.000 | −35% |

These are from old 4-arm design and stale code. Need to re-run after fixes.

## Repo
`/Users/maanavchittireddy/zelo-curriculum-study/`  
GitHub: https://github.com/saimaanav/reranker-curriculum-study

## Key Files
- `src/curricula.py` — 3-arm scheduling (random_cycles, compositional, anti_compositional)
- `src/elo_fit.py` — Bradley-Terry MLE with analytic + autograd Hessian (solid, no changes needed)
- `src/beir_loader.py` — BEIR loading + pool construction (fixed: `rng.shuffle(pool_ids)` prevents BM25 fallback masking)
- `scripts/run_phase2.py` — Main experiment runner (HAS BUGS, see below)
- `configs/fiqa.yaml` — Experiment config (HAS BUGS, see below)
- `results/writeup.md` — Writeup with stale 4-arm results

## Bugs Fixed So Far
1. **TREC-COVID trivial NDCG=1.0**: switched to FiQA (avg 2.2 relevant per 20-doc pool)
2. **BM25 fallback masking (NDCG=1.0)**: pool order was BM25-ranked → all-zero B-T scores → argsort = original order → NDCG=1.0. Fix: `rng.shuffle(pool_ids)` in beir_loader.py ✅
3. **Eval on wrong pools**: BTRatingReranker only scores docs it saw in training. Evaluating on held-out eval_pools → all scores 0.0 → same ranking for all arms. Fix: evaluate on `train_pools` ✅

## Pending Fixes (NOT YET IMPLEMENTED)

### Fix 1 — `scripts/run_phase2.py`: delete stale functions
- Delete `_jaccard()` function
- Delete `_tf_cosine_sim()` function (fake TF bag-of-words cosine, not real semantic similarity)
- Update `build_pair_universe()` to remove calls to both deleted functions

### Fix 2 — `configs/fiqa.yaml`: update arm names
```yaml
# Change from:
arms: [random, difficulty_curriculum, anti_curriculum, compositional_curriculum]
# To:
arms: [random_cycles, compositional, anti_compositional]
```

### Fix 3 — `scripts/run_phase2.py`: real semantic embeddings
- Replace `_tf_cosine_sim` with `sentence-transformers/all-MiniLM-L6-v2`
- Compute embeddings once at startup for all unique docs in training pools
- Use cosine similarity between doc_a and doc_b embeddings as `semantic_sim`

### Fix 4 — `src/curricula.py`: normalize difficulty signals
Current `_difficulty_score` is wrong — no normalization:
```python
# CURRENT (wrong):
def _difficulty_score(p: PairCandidate, alpha=1.0, beta=1.0, gamma=0.5):
    return alpha * p.bootstrap_margin + beta * (1.0 - p.semantic_sim) + gamma * p.lexical_overlap
```
Fix: normalize margin and semantic_sim to [0,1] across all pairs before scoring, then:
```python
difficulty = 0.7 * norm_margin + 0.3 * (1 - norm_semantic_sim)
# drop lexical_overlap (redundant with margin)
```
Normalization must happen at the schedule_* call level (need full pair list to compute min/max).

### Fix 5 — Per-query budget distribution for compositional/anti_compositional
Currently sorts all pairs globally and takes top-N — budget can cluster in a few queries.
Fix: allocate budget proportionally per query (proportional to query's pair count), then sort within each query by difficulty.

## What "in-sample eval" means
`BTRatingReranker` only has ratings for docs it saw during training (the pairs it judged). Evaluating on `train_pools` means we're measuring reranking quality on the same queries we trained on — it's not a held-out test. The writeup should disclose this. It's acceptable for a curriculum scheduling study (the signal is relative across arms, not absolute NDCG), but needs a footnote.

## Current State of curricula.py (has Fix 4 pending)
```python
@dataclass
class PairCandidate:
    query_id: str
    doc_a_id: str
    doc_b_id: str
    bootstrap_margin: float   # |BM25_a - BM25_b|
    lexical_overlap: float    # (BM25_a + BM25_b) / 2
    semantic_sim: float       # embedding cosine similarity (placeholder, not real yet)

def _difficulty_score(p, alpha=1.0, beta=1.0, gamma=0.5):
    # WRONG: unnormalized, wrong coefficients, lexical_overlap included
    return alpha * p.bootstrap_margin + beta * (1.0 - p.semantic_sim) + gamma * p.lexical_overlap

SCHEDULERS = {
    "random_cycles": schedule_random_cycles,
    "compositional": schedule_compositional,
    "anti_compositional": schedule_anti_compositional,
}
```

## After All Fixes
1. Re-run: `python scripts/run_phase2.py --config configs/fiqa.yaml`
2. Update `results/writeup.md` with new 3-arm results
3. Update `README.md` results table with new arm names and numbers
4. Add in-sample eval disclosure footnote to writeup

## Dependencies
- `sentence-transformers` needs to be added to requirements.txt for Fix 3
- All else already installed

## Bradley-Terry / BTRatingReranker
- `src/elo_fit.py` fits B-T MLE via Newton optimizer with analytic + autograd Hessian
- Ratings are used directly as relevance scores (no model training)
- `RatingFitResult` has `.ratings`, `.hessian_analytic`, `.hessian_autograd`, `.converged`
- Hessian computation is already implemented — user asked about "computing the Hessian" but it's already there in Phase 3 analysis context
