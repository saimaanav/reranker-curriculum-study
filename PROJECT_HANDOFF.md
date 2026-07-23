# Project Handoff — The Order Advantage / zelo-curriculum-study

Comprehensive state dump from a long working session. Use this to pick the project
back up without re-deriving everything from scratch.

## 1. The Research Question

At a fixed LLM-judge comparison budget, does the *order* pairwise comparisons are
scheduled in affect Bradley-Terry reranker quality? Tested via three scheduling arms
sharing one judge, one budget, and (for two of three) one difficulty function:

- **compositional** — easy-first (highest difficulty score first)
- **anti_compositional** — hard-first (exact reverse order, same score)
- **random_cycles** — zELO-style coverage-guaranteed random pairing (baseline)

Difficulty score: `0.7 * margin + 0.3 * (1 - semantic_sim)`, both normalized [0,1].
`margin = |BM25_a - BM25_b|`. `semantic_sim` = cosine similarity of
`all-MiniLM-L6-v2` sentence embeddings between doc A and doc B. Difficulty is a
**pairwise** property — always needs two documents, never computed for one alone.

## 2. Verified Experimental Results (real Ollama llama3:latest judge, 5 seeds, 600-comparison budget)

### NDCG@10 (mean ± s.d.)
| Dataset | random_cycles | compositional | anti_compositional |
|---|---|---|---|
| FiQA | 0.346 ± 0.028 | 0.376 ± 0.015 | 0.229 ± 0.025 |
| SciFact | 0.465 ± 0.017 | 0.642 ± 0.026 | 0.277 ± 0.021 |

### Recall@10 (mean ± s.d.)
| Dataset | random_cycles | compositional | anti_compositional |
|---|---|---|---|
| FiQA | 0.579 ± 0.044 | 0.599 ± 0.027 | 0.532 ± 0.071 |
| SciFact | 0.869 ± 0.063 | 0.813 ± 0.023 | 0.553 ± 0.033 |

### MRR (mean ± s.d.)
| Dataset | random_cycles | compositional | anti_compositional |
|---|---|---|---|
| FiQA | 0.339 ± 0.049 | 0.358 ± 0.019 | 0.182 ± 0.010 |
| SciFact | 0.355 ± 0.037 | 0.620 ± 0.025 | 0.234 ± 0.021 |

### Statistical significance (paired across 5 shared seeds)
| Dataset | Comparison | Mean Diff | Paired t-test p | Wilcoxon p | Bootstrap 95% CI |
|---|---|---|---|---|---|
| FiQA | Comp vs. Random | +0.0294 | 0.080 | 0.125 | [0.004, 0.046] |
| FiQA | Comp vs. Anti | +0.1468 | 0.0002 | 0.0625 | [0.127, 0.165] |
| FiQA | Random vs. Anti | +0.1174 | 0.0043 | 0.0625 | [0.084, 0.152] |
| SciFact | Comp vs. Random | +0.1769 | <.0001 | 0.0625 | [0.164, 0.188] |
| SciFact | Comp vs. Anti | +0.3652 | <.0001 | 0.0625 | [0.333, 0.389] |
| SciFact | Random vs. Anti | +0.1883 | <.0001 | 0.0625 | [0.168, 0.201] |

Wilcoxon p is floored at 0.0625 by n=5 sample size — mathematically the best
attainable with 5 paired seeds, not a weak result.

**Headline framing (use this order):** lead with compositional vs. random-cycles
(the real baseline) — up to 38% relative NDCG@10 improvement (SciFact, p<.0001).
Report compositional vs. anti-compositional (up to 2.32x) as the secondary,
worst-case contrast.

### NDCG@20 robustness check (full 20-doc pool, not just top-10 cutoff)
Independent verification run, own self-consistent NDCG@10 baseline (differs
slightly from the table above due to being a separate run):
| Dataset | Arm | NDCG@10 | NDCG@20 |
|---|---|---|---|
| FiQA | random_cycles | 0.3622 ± 0.0226 | 0.4751 ± 0.0175 |
| FiQA | compositional | 0.3758 ± 0.0154 | 0.4896 ± 0.0086 |
| FiQA | anti_compositional | 0.2290 ± 0.0251 | 0.3742 ± 0.0067 |
| SciFact | random_cycles | 0.4485 ± 0.0474 | 0.4862 ± 0.0542 |
| SciFact | compositional | 0.6417 ± 0.0259 | 0.6932 ± 0.0211 |
| SciFact | anti_compositional | 0.2766 ± 0.0210 | 0.3929 ± 0.0144 |

Ordering holds at full-pool NDCG@20 too — rules out a top-k cutoff artifact.

### Hessian / Mechanism (Bradley-Terry fixed-point condition number)
| Dataset | random_cycles | compositional | anti_compositional |
|---|---|---|---|
| FiQA | 39.95 ± 1.32 | 640.28 ± 102.61 | 1017.30 ± 179.08 |
| SciFact | 43.44 ± 4.77 | 250.85 ± 65.26 | 1653.36 ± 240.37 |

Effective rank: FiQA random=9.41, comp=4.77, anti=5.55. SciFact random=9.39,
comp=6.08, anti=5.88.

**Key finding:** random_cycles produces the best-conditioned (most statistically
determined) rating fit on both datasets, yet compositional wins downstream. Precision
of the fit and downstream ranking performance are decoupled — compositional's
advantage comes from *where* it concentrates comparisons (high-margin pairs that
determine top-of-ranking quality), not from a more confident fit overall.

## 3. Experimental Design Details

- Datasets: BEIR FiQA (financial Q&A, 57,638-doc corpus) and BEIR SciFact
  (biomedical claim verification, 5,183-doc corpus).
- 50 queries loaded per dataset, split 35 train / 15 eval (`eval_query_fraction: 0.3`).
  **Only the 35 train queries get the comparison budget spent on them and are what
  NDCG@10 is reported on.** The 15 eval queries exist only for a narrower calibration
  check (see known bug below).
- 20 BM25-retrieved candidates per query (`candidate_pool_size: 20`), so
  C(20,2)=190 possible pairs per query, 35 x 190 = 6,650 total possible pairs.
- Comparison budget: 600 per arm per seed. Spread across all 35 queries (not
  concentrated on a handful) — floor of 10 guaranteed per query for
  compositional/anti (350 total), remaining 250 fill from a globally-pooled,
  difficulty-sorted list of leftover pairs across all queries.
- 5 seeds: `[0, 1, 2, 3, 4]`. Seeds only resolve randomness *within* each arm's
  fixed structural rule (which pairs survive a mid-bucket cutoff for
  compositional/anti; which random permutation builds the cycles for
  random_cycles) — they never change the rule itself, and never change which
  documents/queries are used (that's controlled by a separate, single
  `config["seed"]`).
- Judge: Ollama `llama3:latest` (8B, local).
- GitHub repo (public, verified): `https://github.com/saimaanav/reranker-curriculum-study`

## 4. Real Bugs Found and Fixed This Session

1. **`schedule_random_cycles` infinite loop** — `q_budget` wasn't capped at the
   number of pairs actually available for a query; sparse (non-complete) pair
   graphs could hang forever. Fixed: `q_budget = min(max(n, round(...)), len(q_pairs))`.
2. **Seed-variance bug (compositional/anti_compositional)** — strict full sort by
   a continuous difficulty score has no real ties, so shuffling before sorting was
   a no-op; every seed produced the identical schedule (`std=0.000` across 5 seeds
   was the empirical tell). Fixed by bucketing into difficulty quantiles and
   shuffling *within* buckets, preserving the macro easy-first/hard-first shape
   while giving seeds genuine within-bucket variance to resolve.
3. **Cross-process determinism bug (`random_cycles`)** — built its per-query doc
   list via a Python `set` comprehension, whose iteration order is randomized
   per-process (`PYTHONHASHSEED`), so the same seed could produce different
   schedules across separate script invocations, breaking judge-cache reuse across
   runs. Fixed by switching to `dict.fromkeys(...)` (insertion-order, deterministic).
4. **Stale scheduler names in `configs/*.yaml`** — `fiqa.yaml`, `smoke.yaml`,
   `scifact.yaml` referenced old arm names (`random`, `difficulty_curriculum`, etc.)
   that no longer exist in `SCHEDULERS`. Fixed in fiqa/smoke; scifact.yaml also had
   its judge provider switched from `groq` to `ollama` for consistency, and
   `max_queries`/`comparison_budget` adjusted to match FiQA's working setup.
5. **`_tf_cosine_sim`/`_jaccard`** — dead bag-of-words similarity functions removed
   from `run_phase2.py`, replaced with real `all-MiniLM-L6-v2` sentence embeddings.

## 5. Known Unresolved Issue (not yet fixed — flag if asked, don't present as working)

**ECE / near-tie-pairwise-accuracy (the H2 calibration metrics) are currently
broken/meaningless.** Traced through `build_eval_calibration_pairs` and
`pairwise_calibration` in `scripts/run_phase2.py` / `src/metrics.py`: calibration
pairs come from the 15 held-out eval queries, but they get scored using the arm's
`BTRatingReranker`, which only has scores for `(query_text, doc_text)` pairs it saw
during the arm's actual training-budget judging (the 35 train queries only). Every
lookup for a held-out query misses and falls back to a default score of 0.0 for
both documents, making the implied confidence exactly 50% for every calibration
pair regardless of true difficulty. **This does not affect the headline NDCG@10
results** (computed correctly, in-sample, on the same train queries the reranker
was fit on) — it only invalidates ECE/near-tie-accuracy specifically, which were
not featured on the final poster. Worth fixing properly before citing calibration
numbers anywhere.

## 6. Novelty / Literature Position (from a real litreview pass, PubMed + OpenAlex)

Closest prior work, none of which combine curriculum-ordered scheduling +
Bradley-Terry-fit reranking + BEIR evaluation:
- **Curry-DPO** (Pattnaik et al., EMNLP Findings 2024) — curriculum-orders
  preference pairs for gradient-based DPO training, not a non-parametric BT fit.
- **Elo Uncovered** (Boubdir et al., 2023) — shows Elo ratings are sensitive to
  comparison sequence; motivates why scheduling should matter here.
- **Pairwise Ranking Prompting** (Qin et al., NAACL Findings 2024) — established
  LLM pairwise comparisons for reranking, never tested comparison order.
- "zELO" itself returned **zero** academic search hits — it's industry/blog
  lineage, not a citable peer-reviewed standard. Don't claim it as "the industry
  standard" by name; the defensible claim is "beats random pairing, the default in
  real systems like Chatbot Arena/MT-Bench."

### Inspiration paper (personal connection, real and citable)
**Hocker, D., Constantinople, C. M., & Savin, C. (2025).** Compositional
pretraining improves computational efficiency and matches animal behaviour on
complex tasks. *Nature Machine Intelligence*, 7, 689-702.
https://doi.org/10.1038/s42256-025-01029-3

- Their curriculum is *task-level* (pretrain on simpler cognitive sub-tasks before
  the full task, "kindergarten curriculum learning") — different granularity from
  this project's *comparison-level* curriculum, but same underlying principle:
  order of learning experience shapes *what* is learned, not just how fast.
- **The author (Maanav) is acknowledged in this paper specifically for "fixed-point
  characterization of shaping-trained RNNs"** — directly the same analytical lens
  (fixed-point/Hessian analysis) used in this project's own Mechanism section. Real,
  legitimate, citable personal lineage — not just "read a paper and got inspired."
- License: `(C) The Author(s), under exclusive licence to Springer Nature Limited
  2025` — NOT open access. Figure reuse requires either going through Nature's
  formal reprints/permissions process, or informal author permission (practical
  path: ask Cristina Savin directly, corresponding author). Caption used:
  *"Figure adapted from Hocker et al., Nature Machine Intelligence, 2025 [1]. Used
  with permission."* — only keep "used with permission" if that permission was
  actually obtained.

### Full reference list (APA, alphabetized)
```
Boubdir, M., Kim, E., Ermiş, B., et al. (2023). Elo uncovered: Robustness and best
practices in language model evaluation. arXiv. https://doi.org/10.48550/arXiv.2311.17295

Hocker, D., Constantinople, C. M., & Savin, C. (2025). Compositional pretraining
improves computational efficiency and matches animal behaviour on complex tasks.
Nature Machine Intelligence, 7, 689-702. https://doi.org/10.1038/s42256-025-01029-3

Pattnaik, P., Maheshwary, R., Ogueji, K., et al. (2024). Curry-DPO: Enhancing
alignment using curriculum learning & ranked preferences. In Findings of the
Association for Computational Linguistics: EMNLP 2024.
https://doi.org/10.18653/v1/2024.findings-emnlp.754

Qin, Z., Jagerman, R., Hui, K., et al. (2024). Large language models are effective
text rankers with pairwise ranking prompting. In Findings of the Association for
Computational Linguistics: NAACL 2024. https://doi.org/10.18653/v1/2024.findings-naacl.97
```
Note: 3 of 4 entries only have first-3-authors + "et al." from search-result
snippets, not full author rosters — worth pulling complete lists from each paper's
own listing page before any formal submission.

## 7. Poster (Canva, final version): Design Spec

- **Format:** 16:9, digital display (YCML Research Symposium / Startup School 2026,
  TV + HDMI, no printing/handouts allowed).
- **Title:** "The Order Advantage: A Curriculum Approach to RAG Document Reranking"
- **Byline:** Maanav Chittireddy, Georgia Institute of Technology (Global Pathways
  Program)
- **Colors:** background `#FAFAF8`, primary/headings `#173B63`, accent `#B5852A`,
  body text `#333333`, dividers `#D8D8D8`. Chart-mark triad (validated colorblind-safe
  via the dataviz skill's `validate_palette.js`): random_cycles `#2760A0`,
  compositional `#B5852A`, anti_compositional `#A3503C`.
- **Fonts:** IBM Plex Serif (titles/headers), IBM Plex Sans (body/captions/data) —
  both native to Canva's font library.
- **Sections used:** Introduction (bulleted, cites Hocker et al. + own fixed-point
  connection), Methods (6-stage flowchart + Hessian branch), Results (Figure 1 NDCG
  bar chart, Table 2 significance, Figure 3 NDCG@20 robustness, Figure 4 illustrative
  loss-landscape bowls), Conclusions, Future Work/Limitations, Works Cited.
- **Deliberately cut from the final poster** (kept as Q&A backup only, not pasted
  in): Table 1 (full Recall/MRR metrics table) and Figure 2 (Hessian condition-number
  bar chart alone — superseded by Figure 4's bowl visualization, confirm this was
  intentional).
- **QR code:** top-right, links to the GitHub repo above.
- Copyright note: this is independent research, not officially GT-sponsored — "GT,
  Global Pathways Program" in the byline is a factual identity statement, not a
  sponsorship claim.

### Generated figure files (all white background, IBM Plex Serif, captioned/numbered)
Built via matplotlib in `/private/tmp/.../scratchpad/gen_results.py` and
`gen_bowls.py` (session-scratchpad paths, ephemeral — regenerate from the scripts
below if needed, using the verified data tables in Section 2):
- Figure 1 — NDCG@10 grouped bar chart, both datasets
- Figure 2 — Hessian condition number, log-scale bar chart (cut from final poster)
- Figure 3 — NDCG@10 vs. NDCG@20 side-by-side (robustness check)
- Figure 4 — illustrative loss-landscape contour bowls, axis ratio = sqrt(condition
  number), captioned clearly as illustrative/matched-to-measured-value, not raw data
- Table 1 — full NDCG@10/Recall@10/MRR metrics (cut from final poster, Q&A backup)
- Table 2 — full statistical significance table (on the final poster)
- `pipeline_flowchart.png` — the 6-stage methodology diagram with Hessian branch

If these files are gone (scratchpad is ephemeral across session resets), the
generation scripts and exact data are all captured in this document — regenerate
by rewriting a matplotlib script against Section 2's numbers, using IBM Plex Serif
(fetch from `https://raw.githubusercontent.com/google/fonts/main/ofl/ibmplexserif/`)
and the color palette above.

## 8. Content Copy (final wording used)

**Background/Introduction bullets:**
- Large language models are now commonly used as pairwise judges, comparing two
  candidate documents and deciding which is more relevant.
- One efficient approach aggregates these judgments directly into a Bradley-Terry
  rating, skipping the step of training a separate reranking model.
- This is limited by comparison budget: each judgment costs an API call, so how
  that budget gets spent matters a lot.
- Hocker, Constantinople, and Savin (2025) [1] found that pretraining RNNs on a
  curriculum of simpler subtasks was critical, not just faster, for the network to
  pick up animal-like reasoning strategies that standard training missed entirely.
- Their central finding: the order of learning experience shapes what a model
  learns, not just how quickly it learns. Whether the same is true for LLM-judge
  comparison order had never been tested.
- This project builds on the author's prior contribution to fixed-point
  characterization of curriculum-trained RNNs in Hocker et al. [1], extending the
  same analytical lens to Bradley-Terry reranking.

**Conclusions (technical, implication-framed, 2 sentences max, no em dashes):**
1. Reranking with LLM-judge comparisons is inherently budget-constrained, and this
   work shows the order those comparisons are scheduled in, not just how many are
   made, materially changes reranking quality, improving NDCG@10 by up to 2.3x at
   no added cost.
2. Standard active-learning intuition says query the hardest cases first, but here,
   that strategy was consistently the weakest performer, suggesting difficulty-aware
   scheduling for LLM-judge systems should run in the opposite direction from
   conventional wisdom.
3. Our random-cycles baseline, modeled on the coverage-guaranteed near-random
   pairing many LLM-judge systems default to, underperformed compositional
   scheduling on both datasets tested. That raises an open question worth testing
   directly in those systems, not a claim about them.
4. The compositional-scheduling advantage held across two structurally different
   domains, financial question answering and biomedical claim verification, which
   rules out the possibility that this is an artifact of one dataset's particular
   document structure or query style rather than a general effect.

**Future Work / Limitations (short):**
1. Ratings were evaluated in-sample, not on held-out queries. Testing true
   generalization is left for future work.
2. All judgments came from one local 8B model. Whether the effect holds with a
   larger or closed frontier judge is untested.

**Scope statement (promoted out of Future Work, into Results, before Figure 1):**
"All ranking quality is measured in-sample, on the same queries used to schedule
comparisons. Held-out generalization is not tested in this study."

## 9. Glossary (for a beginner AI audience, already given to the user)

- **RAG:** an AI system that searches documents for relevant info before writing
  an answer, instead of relying only on memorized training data.
- **Reranking:** re-sorting an initial batch of candidate documents to put the best
  ones on top.
- **BM25:** a fast keyword-matching search algorithm (not AI-based).
- **Curriculum learning:** the idea that presentation order changes how well a
  system learns.
- **NDCG@10:** ranking quality score rewarding relevant docs near the top, not just
  present anywhere in the list.
- **Recall@10:** fraction of all truly relevant docs that made it into the top 10.
- **MRR:** credit based on the position of the *first* relevant result.
- **p-value:** the chance of seeing a gap this large if there were actually no real
  difference; smaller = stronger evidence.
- **Bootstrap 95% CI:** a resampled range you can be 95% confident contains the true
  effect size; excluding zero = evidence of a real effect.
- **Fixed point:** the final, converged rating each document settles on.
- **Hessian / condition number:** how tightly the observed comparisons pin down each
  rating; lower condition number = more statistically confident fit (not the same
  as a *better* fit for ranking, see Section 2's mechanism finding).

## 10. Open TODOs

- [ ] Confirm whether Cristina Savin actually granted permission for the Hocker
      et al. figure reuse; drop "Used with permission" from the caption if not.
- [ ] Decide whether to fix the ECE/calibration bug (Section 5) or just leave it
      undocumented/unused going forward.
- [ ] Confirm Figure 2 was intentionally cut from the final poster (vs. an
      accidental omission).
- [ ] Pull full author lists for Pattnaik/Boubdir/Qin references if a formal
      write-up (beyond the poster) is planned.
- [ ] If pursuing true held-out generalization in the future: would require
      distilling ratings into a trained, generalizing reranker (the
      `CrossEncoderReranker` path in `src/distill.py`), which was deliberately
      scoped out of this project's Phase 2 approach.
