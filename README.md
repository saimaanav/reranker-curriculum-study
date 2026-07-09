# zelo-curriculum-study

**Does the order you make pairwise comparisons in matter for reranker training?**

This project tests whether difficulty-scheduled curricula improve Bradley-Terry reranker training under a fixed LLM judge budget — inspired by ZeroEntropy's [zELO method](https://arxiv.org/abs/2509.12541). Independent open-benchmark study; not a reproduction of ZeroEntropy's production models.

---

## Result

On FiQA (BEIR), 50 queries, 300 judge calls, 5 seeds:

| Scheduling arm | NDCG@10 | vs. random |
|---|---|---|
| **compositional_curriculum** | **0.388 ± 0.020** | **+14%** |
| difficulty_curriculum | 0.369 ± 0.000 | +8% |
| random (baseline) | 0.340 ± 0.032 | — |
| anti_curriculum | 0.221 ± 0.000 | −35% |

Compositional curriculum beats random **5/5 seeds**. Anti-curriculum loses to random **5/5 seeds**. At 25% of the budget (75 comparisons), curriculum arms already match random's final score — scheduling is more sample-efficient.

See [`results/writeup.md`](results/writeup.md) for the full findings.

---

## How it works

Each scheduling arm gets the same fixed budget of LLM judge calls (pairwise: "is doc A or B more relevant to this query?"). The arms differ only in *which pairs* they spend the budget on:

- **random**: uniform random pair selection
- **difficulty_curriculum**: easiest pairs first (highest BM25 margin between docs)
- **anti_curriculum**: hardest pairs first (lowest BM25 margin)
- **compositional_curriculum**: balances lexical overlap + semantic similarity signals

Judged pairs are fed into Bradley-Terry MLE to produce per-document relevance ratings, which are used directly as reranker scores (no model training required).

---

## Quickstart

Requires [Ollama](https://ollama.com) with Llama 3 pulled — free, runs fully locally.

```bash
ollama pull llama3

git clone <this repo> && cd zelo-curriculum-study
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# download FiQA
python scripts/download_data.py --dataset fiqa

# run the experiment (~10 min on M-series Mac)
python scripts/run_phase2.py --config configs/fiqa.yaml
```

To use Groq instead of Ollama, set `provider: groq` in `configs/fiqa.yaml` and export `GROQ_API_KEY`.

**Smoke test (no network, no API key, ~1s):**
```bash
python scripts/run_phase2.py --config configs/smoke.yaml --smoke
```

---

## Repo structure

```
src/
  beir_loader.py      # BEIR dataset loading + candidate pool construction
  curricula.py        # Scheduling arms
  elo_fit.py          # Bradley-Terry MLE fitting
  judge.py            # LLM judge client (Ollama / Groq / OpenRouter)
  metrics.py          # NDCG@10, Recall, MRR, ECE, near-tie accuracy
scripts/
  run_phase2.py       # Main experiment script
configs/
  fiqa.yaml           # Experiment config
results/
  phase2/fiqa/        # Raw JSON results
  writeup.md          # Full findings
```

---

## Tests

```bash
python -m pytest tests/ -v
```

Cross-checks Bradley-Terry MLE against `choix` on synthetic data with known ground-truth ratings.
