# The Order Advantage: Curriculum Scheduling for LLM-Judge Reranking

**At a fixed LLM-judge comparison budget, does the *order* pairwise comparisons are made in change reranker quality?**

Independent open-benchmark study (BEIR FiQA + SciFact), inspired by ZeroEntropy's
[zELO method](https://arxiv.org/abs/2509.12541) — not a reproduction of ZeroEntropy's
production models. Presented as a poster, *"The Order Advantage: A Curriculum Approach
to RAG Document Reranking,"* at the YCML Research Symposium / Startup School 2026.

---

## TL;DR

Three scheduling arms share one judge, one budget, and one difficulty function. They
differ only in *which* pairs get judged first under a fixed 600-comparison budget:

- **compositional** — spend the budget on the easiest pairs first
- **anti_compositional** — spend the budget on the hardest pairs first (same score, reversed)
- **random_cycles** — zELO-style coverage-guaranteed random pairing (the real baseline)

**Compositional wins on both datasets, by a wide and statistically significant margin.**
Querying the hardest pairs first — the standard active-learning intuition — was
consistently the *worst* strategy.

| Dataset | random_cycles | compositional | anti_compositional |
|---|---|---|---|
| FiQA NDCG@10 | 0.346 ± 0.028 | **0.376 ± 0.015** | 0.229 ± 0.025 |
| SciFact NDCG@10 | 0.465 ± 0.017 | **0.642 ± 0.026** | 0.277 ± 0.021 |

Up to **38% relative improvement over random pairing** (SciFact, paired t-test
p < .0001) and up to **2.3x over hardest-first scheduling**, at no added cost — same
judge, same budget, same difficulty function. Full statistics (paired t-test, Wilcoxon,
bootstrap CI, 5 seeds) in [Results](#results) below.

---

## Why order shouldn't matter, and why it still does

Bradley-Terry maximum-likelihood fitting is **order-invariant**: given a fixed set of
judged comparisons, the fit is identical no matter what sequence they're processed in.
So "scheduling order" isn't a sequence effect inside the optimizer — it's a **data
selection problem under a fixed budget**. With only 600 of ~6,650 possible pairs
affordable per run, the scheduling policy decides *which* 600 comparisons the model
ever gets to see. Compositional and anti-compositional are, in effect, two different
(mostly non-overlapping) datasets built from opposite ends of the same difficulty
spectrum — not the same dataset processed in different order.

## The mechanism: a more confident fit is not a better ranking

The surprising part isn't just that compositional wins — it's *why*. Looking at the
Hessian of the Bradley-Terry fixed point (condition number = how tightly the observed
comparisons pin down the fit; lower = more statistically confident):

| Dataset | random_cycles | compositional | anti_compositional |
|---|---|---|---|
| FiQA | 39.95 ± 1.32 | 640.28 ± 102.61 | 1017.30 ± 179.08 |
| SciFact | 43.44 ± 4.77 | 250.85 ± 65.26 | 1653.36 ± 240.37 |

**Random cycles produces the best-conditioned (most confident) fit on both datasets —
yet compositional still wins downstream.** Fit confidence and ranking quality are
decoupled: compositional's advantage comes from *where* it concentrates its budget
(high-margin, judge-reliable comparisons that determine top-of-ranking correctness),
not from a more statistically confident fit overall. This echoes Hocker, Constantinople
& Savin (2025, *Nature Machine Intelligence*) — curriculum ordering changes *what* a
system learns, not just how confidently it learns it.

---

## How the scheduling works

Every candidate pair `(doc_a, doc_b)` within a query gets a difficulty score:

```
difficulty = 0.7 * margin + 0.3 * (1 - semantic_sim)
```

- `margin` = `|BM25_a - BM25_b|`, min-max normalized to [0,1] — large gap = easy
- `semantic_sim` = cosine similarity of `all-MiniLM-L6-v2` sentence embeddings between
  the two docs, normalized to [0,1] — high similarity = confusingly close = hard, hence
  inverted

**compositional / anti_compositional:** all pairs (pooled across every query) are
sorted by difficulty and bucketed into 10 quantile bins; bins are read easiest-first or
hardest-first depending on the arm, with a random shuffle *within* each bin (seeded, for
reproducible variance across runs). Each of the 35 train queries is first guaranteed a
floor of 10 pairs from its own front of that ordering, then the remaining budget is
filled from a globally-pooled leftover list in the same difficulty order — so queries
with more extreme-difficulty pairs draw disproportionately more of the fill budget.

**random_cycles:** no difficulty scoring at all. Each query gets a proportional share
of the budget (`round(budget * query's pair count / total pairs)`, floored at its doc
count so every doc is touched at least once, capped at its available pair count so the
loop can't spin forever on a sparse query). That quota is filled by building random
"ring cycles" over the query's docs — each doc paired with its neighbors in a shuffled
permutation, guaranteeing every doc appears at least twice — reshuffling and adding new
cycles until the quota is met.

See [`src/curricula.py`](src/curricula.py) for the full implementation of all three arms.

---

## Repo structure

```
src/
  curricula.py         # The three scheduling arms (this is the core of the study)
  beir_loader.py        # BEIR dataset loading + candidate pool construction
  elo_fit.py             # Bradley-Terry MLE fitting
  judge.py                # LLM judge client (Ollama / Groq / OpenRouter)
  metrics.py              # NDCG@10, Recall@10, MRR, ECE, near-tie accuracy
scripts/
  run_phase2.py                    # Main experiment: schedule -> judge -> fit -> score
  run_phase3.py                    # Hessian / condition-number mechanism analysis
  run_phase3_hessian_only.py       # Hessian analysis only, reusing existing judge results
  check_ndcg_at_full_pool.py       # NDCG@20 robustness check (rules out top-k cutoff artifact)
configs/
  fiqa.yaml, scifact.yaml   # Full experiment configs (Ollama llama3, 5 seeds, 600-budget)
  smoke.yaml                # Offline, no-network, no-API-key smoke test
results/
  phase2/{fiqa,scifact}/    # Per-dataset results tables (raw JSON gitignored — regenerate via scripts)
tests/
  test_curricula.py     # Scheduling-arm unit tests (coverage guarantees, budget bounds, determinism)
PROJECT_HANDOFF.md       # Full experimental log: results, bugs fixed, poster copy, open TODOs
```

---

## Quickstart

Requires [Ollama](https://ollama.com) with Llama 3 pulled — free, runs fully locally.

```bash
ollama pull llama3

git clone https://github.com/saimaanav/reranker-curriculum-study.git
cd reranker-curriculum-study
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# download a BEIR dataset (fiqa or scifact)
python scripts/download_data.py --dataset fiqa

# run the full experiment (5 seeds, 600-comparison budget)
python scripts/run_phase2.py --config configs/fiqa.yaml
```

To use Groq instead of Ollama, set `provider: groq` in the config and export
`GROQ_API_KEY`.

**Smoke test (no network, no API key, ~1s):**
```bash
python scripts/run_phase2.py --config configs/smoke.yaml --smoke
```

**Tests:**
```bash
python -m pytest tests/ -v
```
Cross-checks Bradley-Terry MLE against `choix` on synthetic data with known
ground-truth ratings, plus unit tests for all three scheduling arms.

---

## Known limitations

- All ranking quality is measured **in-sample**, on the same queries used to schedule
  comparisons. Held-out generalization is not tested in this study.
- All judgments come from one local 8B model (Ollama llama3). Whether the effect holds
  with a larger or closed frontier judge is untested.
- The eval-query calibration metrics (ECE, near-tie pairwise accuracy) have a known
  scoring bug — see `PROJECT_HANDOFF.md` §5 — and should not be cited.

See [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md) for the full experimental log, including
every bug found and fixed, statistical methodology, literature positioning, and open
TODOs.
