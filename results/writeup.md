# zelo-curriculum-study: Does Comparison Scheduling Matter for Pairwise Reranking?

**Inspired by ZeroEntropy's zELO method.** This project tests whether difficulty-scheduled pairwise curricula improve reranker training under a fixed comparison budget.

---

## Setup

| | |
|---|---|
| **Dataset** | FiQA (financial QA, BEIR benchmark) |
| **Corpus** | 57,638 documents |
| **Queries** | 50 (35 train / 15 eval) |
| **Candidate pool** | k=20 per query |
| **Comparison budget** | 300 total LLM judge calls |
| **Judge** | Llama 3 8B via Ollama (local, no API cost) |
| **Reranker** | Bradley-Terry MLE ratings used directly as relevance scores |
| **Seeds** | 5 (0–4) |

---

## Scheduling Arms

| Arm | Strategy |
|---|---|
| `random` | Pairs sampled uniformly at random |
| `difficulty_curriculum` | Easiest pairs first (highest BM25 score margin) |
| `anti_curriculum` | Hardest pairs first (lowest BM25 score margin) |
| `compositional_curriculum` | Balances lexical overlap + semantic similarity signals |

---

## Results (NDCG@10, 5 seeds)

| Arm | NDCG@10 | MRR | vs. random |
|---|---|---|---|
| **compositional_curriculum** | **0.388 ± 0.020** | **0.360** | **+14%** |
| difficulty_curriculum | 0.369 ± 0.000 | 0.366 | +8% |
| random | 0.340 ± 0.032 | 0.330 | baseline |
| anti_curriculum | 0.221 ± 0.000 | 0.133 | −35% |

**Compositional curriculum beats random in all 5/5 seeds. Anti-curriculum loses to random in all 5/5 seeds.**

---

## Learning Curves (seed=0, NDCG@10 at 25/50/75/100% of budget)

| Comparisons | random | difficulty | anti | compositional |
|---|---|---|---|---|
| 75  | 0.231 | 0.313 | 0.267 | 0.306 |
| 150 | 0.268 | 0.311 | 0.235 | 0.330 |
| 225 | 0.261 | 0.381 | 0.228 | 0.351 |
| 300 | 0.281 | 0.369 | 0.221 | 0.374 |

At just 75 comparisons (25% of budget), both curriculum arms already match or exceed random's final score at 300 comparisons. Curriculum scheduling is more sample-efficient.

---

## Key Findings

**1. Pair selection order matters significantly.**
A 35% NDCG gap separates the best (compositional, 0.388) and worst (anti, 0.221) arms using identical judge budgets and the same BM25 retrieval baseline. The only difference is which pairs the budget is spent on.

**2. Easy-first isn't enough — balance wins.**
Pure difficulty curriculum (+8%) is outperformed by the compositional arm (+14%), which mixes lexical and semantic difficulty signals. Spending the whole budget on obvious comparisons leaves value on the table.

**3. Anti-curriculum actively degrades quality.**
Starting with near-tie pairs the judge struggles to distinguish injects noise into early B-T estimates. NDCG degrades monotonically (0.267 → 0.221) as budget grows — more comparisons make it worse, not better.

**4. Sample efficiency is the main gain.**
Curriculum arms reach random's final NDCG in ~25% of the budget. The practical implication: with a tight judge call budget, curriculum scheduling recovers significant ranking quality that random wastes on uninformative pairs.

---

## Method Notes

- **No model training.** Bradley-Terry ratings are fitted via MLE from judged pairs and used directly as document scores — no cross-encoder fine-tuning.
- **Free inference.** All comparisons made locally via Ollama (Llama 3 8B). Full experiment runs in ~10 minutes on an M-series Mac.
- **Pair structure matters for dataset choice.** TREC-COVID was tested first but discarded — its qrels labeled nearly all k=20 pool docs as relevant, making NDCG trivially 1.0 for all arms. FiQA's sparse relevance (avg 2.2 relevant per 20-doc pool) makes the ranking problem genuinely hard.

---

## Reproduction

```bash
git clone <this repo>
pip install -r requirements.txt
# pull FiQA via BEIR
python scripts/download_data.py --dataset fiqa
# run experiment
python scripts/run_phase2.py --config configs/fiqa.yaml
```

Requires [Ollama](https://ollama.com) with `llama3` pulled (`ollama pull llama3`).

---

## Repo Structure

```
zelo-curriculum-study/
├── src/
│   ├── beir_loader.py      # BEIR dataset loading + candidate pool construction
│   ├── curricula.py        # Scheduling arms (random, difficulty, anti, compositional)
│   ├── elo_fit.py          # Bradley-Terry MLE fitting
│   ├── judge.py            # LLM judge client (Ollama / Groq / OpenRouter)
│   └── metrics.py          # NDCG@10, Recall, MRR, ECE, near-tie accuracy
├── scripts/
│   └── run_phase2.py       # Main experiment script
├── configs/
│   └── fiqa.yaml           # Experiment config
└── results/
    └── phase2/fiqa/        # Raw JSON results + this writeup
```
