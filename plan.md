# plan.md — Difficulty-Scheduled Comparison Curricula for zELO-Style Reranker Training

## 1. One-line thesis

Scheduling pairwise comparisons by Elo-margin difficulty (large-margin battles first, near-ties last) makes Bradley-Terry/Thurstone rating fits converge to a better-conditioned fixed point under a fixed comparison budget, and that conditioning predicts downstream reranker calibration and near-tie NDCG better than random sampling.

## 2. Repo structure

```
zelo-curriculum-study/
├── plan.md
├── README.md                    # exact repro commands, filled in as phases land
├── requirements.txt              # pinned versions
├── configs/
│   ├── smoke.yaml                # tiny config for --smoke runs
│   ├── fiqa.yaml
│   └── scifact.yaml
├── data/
│   ├── raw/                      # BEIR downloads (gitignored)
│   └── cache/                    # cached LLM judge calls, keyed by hash(query,docA,docB,judge_model)
├── src/
│   ├── elo_fit.py                 # Bradley-Terry/Thurstone MLE, exposes rating vector + Hessian at fixed point
│   ├── comparison_difficulty.py   # margin difficulty, judge flip-rate, cross-query bias signals
│   ├── judge.py                   # Anthropic API pairwise judge wrapper + disk cache
│   ├── beir_loader.py              # BEIR (FiQA, SciFact) candidate set construction
│   ├── curricula.py                # random / difficulty-scheduled / anti-curriculum / compositional schedulers
│   ├── distill.py                  # rating -> pointwise cross-encoder distillation (sentence-transformers, later Qwen3-0.6B LoRA via PEFT)
│   ├── metrics.py                  # NDCG@10, Recall, MRR, ECE, reliability curves, near-tie slicing
│   ├── hessian_analysis.py         # eigenspectrum/conditioning of fixed-point Hessian; score-distribution geometry tracking
│   └── utils.py                    # seeding, config loading, W&B optional wrapper
├── scripts/
│   ├── run_phase1.py                # generate comparisons, fit ratings, difficulty analysis
│   ├── run_phase2.py                # curriculum vs baseline training sweep (multi-seed)
│   └── run_phase3.py                # Hessian eigenspectrum + H3 correlation analysis
├── tests/
│   ├── test_elo_fit.py              # correctness cross-check vs `choix` on synthetic data
│   ├── test_comparison_difficulty.py
│   └── test_metrics.py
└── results/
    ├── phase1/
    ├── phase2/
    └── phase3/                     # tables, learning curves, key figure
```

## 3. Datasets

- **FiQA** (finance, BEIR) — Phase 1 primary domain.
- **SciFact** (STEM/scientific claims, BEIR) — Phase 1 second domain, cross-query bias check.
- Candidate sets: top-k (k≈20–50) BM25 or existing dense retriever results per query, so pairwise judging budget stays bounded and near-tie pairs actually exist within the candidate pool.
- Interface designed so a code-retrieval BEIR-style set (e.g., CodeSearchNet-derived) can be added later without changing `elo_fit.py` or `curricula.py`.

## 4. Rating fit math (from zELO / Bradley-Terry-Thurstone)

- Each query has its own rating vector over its candidate documents (ratings are not shared across queries directly).
- Pairwise outcome model: `P(A beats B) = sigmoid(r_A - r_B)` (Bradley-Terry / logistic) as primary; Thurstone (probit) as a config-selectable alternative.
- MLE via full-batch gradient descent / Newton's method in PyTorch (autograd), not `choix`. `choix` is used only in `tests/test_elo_fit.py` as a correctness cross-check on synthetic pairwise data with a known ground-truth rating vector.
- **First-class outputs required by the design doc:** `elo_fit.py` must return (a) converged rating vector `r*`, (b) the Hessian `H = ∂²(-log L)/∂r²` at `r*`, computed via `torch.autograd.functional.hessian` (or manual analytic Bradley-Terry Hessian, which has a closed form: `H_ij = -w_ij * p_ij * (1-p_ij)` off-diagonal, row-sum on diagonal — implement both, cross-check against each other).
- Cross-query bias term: an additive per-domain/per-query offset fit jointly (or via a second regression pass) so ratings are comparable across FiQA vs SciFact — mirrors zELO's cross-query calibration.

## 5. Comparison difficulty definition

- Primary signal: `difficulty(A,B) = |r_A - r_B|` at current rating estimate (small margin = hard/near-tie).
- Secondary signals (for validation, not scheduling):
  - Judge flip-rate: repeat the same pairwise judge call N times (e.g., N=3) with temperature > 0 or resampled prompt order; disagreement rate is a difficulty proxy.
  - Cross-query bias magnitude: how much the fitted per-query offset shifts a pair's apparent difficulty.
- `comparison_difficulty.py` scores any (query, docA, docB) pair set and exposes both an initial-margin estimate (before ratings are known, e.g. from a cheap first-pass embedding similarity or a small seed round of random battles) and a running/updated margin as the rating fit iterates — the curriculum needs a way to bootstrap difficulty before ratings converge.

## 6. Pairwise judge

- **$0 budget constraint**: no paid API usage. Oracle ensemble sourced from **OpenRouter's free-tier models** (e.g. `meta-llama/llama-3.1-8b-instruct:free`, `google/gemini-flash-*:free` or similar free-tier IDs available at run time — resolved to concrete model IDs in `configs/*.yaml` since OpenRouter's free catalog changes), each prompted with query + doc A + doc B, asked to pick the more relevant one (plus optional confidence). Ensembling two free-tier judges gives a majority-vote label and a natural inter-judge-disagreement signal (feeds into `comparison_difficulty.py`'s flip-rate proxy alongside repeated same-judge calls).
- Free-tier models carry lower judge quality and stricter rate limits than Claude 3.5 Sonnet/Gemini 1.5 Pro — the writeup must flag this as a limitation (judge noise floor is higher than zELO's production oracle), and the comparison budget `B` must be sized to fit free-tier rate limits (likely means sequential/throttled requests, planned into `judge.py`'s retry/backoff logic).
- `judge.py` is a thin OpenRouter client (single API surface, swappable model IDs) rather than calling Anthropic/Google APIs directly — keeps model choice a config value, not code.
- Every call cached to disk (`data/cache/`) keyed by a stable hash of (model, prompt version, query id, doc A id, doc B id, order) — order-swapped calls cached separately to detect position bias, and both judges' calls for the same pair are cached independently.
- Deterministic: cache means repeated runs are free and reproducible; only new pairs hit the API.
- **Dev-time debugging only**: a local Llama 3.2 1B via Ollama is used purely for unit-test/logic debugging convenience (e.g., quick sanity-check completions while iterating on `judge.py`'s prompt plumbing) — never used as an oracle judge or anywhere in reported results.

## 7. Experiments

### Phase 1 — comparison difficulty in Elo coordinates
1. Fit ratings from an initial random comparison set per query (FiQA, SciFact separately).
2. Compute margin-difficulty for all judged pairs from Phase 1's fit.
3. Analysis script: correlate margin-difficulty with (a) judge flip-rate on repeated calls, (b) reranker pointwise error after a baseline distillation — establishing that margin is a real difficulty signal before it's used to schedule anything.

### Phase 2 — curriculum vs baseline training
- Fixed total comparison budget `B` (same across all arms).
- Arms (≥3 seeds each):
  1. **Random** — uniform random pair sampling until budget exhausted.
  2. **Difficulty-scheduled curriculum** — large-margin pairs first (using an initial bootstrap round to estimate margins), progressively shifting to near-tie pairs as budget is spent.
  3. **Anti-curriculum** — near-tie pairs first (ablation, tests whether ordering direction matters or just exposure).
  4. **Compositional sub-skill curriculum** — ordered lexical-overlap pairs → semantic-similarity pairs → instruction-following pairs → hard-distractor pairs (sophisticated variant, needs pair categorization heuristics per BEIR domain).
- For each arm/seed: fit ratings under that comparison schedule, distill into pointwise cross-encoder (sentence-transformers cross-encoder baseline; Qwen3-0.6B LoRA via PEFT if compute allows), evaluate on held-out queries.
- Metrics: NDCG@10, Recall@k, MRR, **comparisons-to-target** (number of comparisons needed to reach a fixed NDCG@10/ECE threshold — H1), ECE + reliability curves (H2), near-tie-slice NDCG (queries/pairs bucketed by Phase 1 margin-difficulty — H2).
- Output: results table (mean ± std across seeds), learning curves (metric vs comparisons spent), seed variance plot.

### Phase 3 — dynamics analysis of the rating fixed point
1. At convergence, compute Hessian eigenspectrum for curriculum vs random comparison graphs at equal budget (condition number = λ_max/λ_min, spectral gap, eigenvalue histogram).
2. Correlate fixed-point conditioning (per query, per seed) against Phase 2's downstream calibration (ECE) and near-tie NDCG — test H3 via correlation/regression across seeds and queries.
3. Cross-check: track cross-encoder score-distribution geometry over training steps (score separation between relevant/irrelevant, calibration drift, effective dimensionality via PCA of penultimate-layer embeddings).
4. Produce the key figure: two-panel — (left) curriculum vs random on calibration + near-tie NDCG at equal budget; (right) Hessian conditioning comparison, positioned as the mechanism explaining the left panel.

## 8. Metrics summary

| Hypothesis | Metric | Computed in |
|---|---|---|
| H1 | Comparisons-to-target (NDCG@10, ECE thresholds) | Phase 2 |
| H1 | NDCG@10, Recall, MRR at fixed budget | Phase 2 |
| H2 | ECE, reliability curves | Phase 2 |
| H2 | Near-tie-slice NDCG (sliced by Phase 1 margin) | Phase 2 |
| H3 | Hessian eigenspectrum, condition number | Phase 3 |
| H3 | Correlation: conditioning vs ECE / near-tie NDCG | Phase 3 |

## 9. Reproducibility & constraints checklist

- Single GPU, fixed modest comparison budget (exact number set once FiQA/SciFact candidate pool sizes are known — target order of magnitude ~5k–20k judged pairs total across arms).
- 3+ seeds on every reported result; seeds control comparison sampling order, model init, and LoRA init.
- All deps pinned in `requirements.txt` (exact versions, hashes where feasible).
- All LLM judge calls cached; cache is part of the artifact so full reruns don't re-call the API.
- Every script (`run_phase1.py`, `run_phase2.py`, `run_phase3.py`) supports `--smoke` running on a tiny sample (e.g., 5 queries, 20 pairs, 1 seed, 1 training step) end-to-end before any full run.
- W&B logging optional and offline-capable (`WANDB_MODE=offline` or a `--no-wandb` flag).
- `choix` used only in tests as a correctness cross-check against our own PyTorch MLE — never load-bearing.
- Infra mapping: Phase 1 pairwise judging → OpenRouter **free-tier** models ensemble (cached, $0 cost, rate-limit-aware backoff); Phase 1 rating fit/Hessian, Phase 2 training, Phase 3 dynamics → local/Colab single GPU, custom PyTorch/SciPy; dev-only debugging → local Ollama Llama 3.2 1B (never a scored judge or reported result).
- Hard $0 API budget: no paid model calls anywhere in the pipeline. If a free-tier model becomes unavailable/rate-limited mid-run, `judge.py` must fail loudly (not silently substitute a paid model) so cost never creeps in unnoticed.
- README will list exact commands (with config flags and expected runtime) to reproduce every reported number.
- No overclaiming: writeup will explicitly state this is an independent open-benchmark study, not a reproduction of ZeroEntropy's production zerank models.

## 10. Definition of done

- Clean repo matching structure above, passing `--smoke` on all three phase scripts.
- 4–6 page writeup (`results/writeup.md` or PDF) with the single key figure (Phase 3, item 4 above).
- Honest reporting: if H1/H2/H3 fail or show null results, that is reported plainly with the Hessian-based failure analysis, not hidden.

## 11. Open questions for review before Phase 1 starts

1. Candidate pool size `k=20` candidates/query, confirmed. Comparison budget `B` chosen so each arm gets ~5–10 comparisons/query average, adjustable after a smoke run shows free-tier rate-limit throughput (since $0 budget means throughput, not cost, is the binding constraint).
2. ~~Judge model choice~~ — resolved: OpenRouter free-tier model ensemble as the oracle (cached calls, $0 cost), with Ollama Llama 3.2 1B reserved for dev-only debugging.
3. ~~Cross-encoder base model~~ — resolved: keep `cross-encoder/ms-marco-MiniLM-L-6-v2` as the sentence-transformers base for Phase 2, trained from scratch on our fitted ratings.
