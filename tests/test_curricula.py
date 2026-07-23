from src.curricula import (
    PairCandidate,
    schedule_anti_compositional,
    schedule_compositional,
    schedule_random_cycles,
)


def _make_pairs():
    # margins: 1.0 (easy) .. 0.01 (near-tie), all normalized to [0,1].
    # semantic_sim held constant so difficulty score is monotonic in margin alone.
    return [
        PairCandidate("q1", "a0", "b0", bootstrap_margin=1.0, semantic_sim=0.5),
        PairCandidate("q1", "a1", "b1", bootstrap_margin=0.7, semantic_sim=0.5),
        PairCandidate("q1", "a2", "b2", bootstrap_margin=0.4, semantic_sim=0.5),
        PairCandidate("q1", "a3", "b3", bootstrap_margin=0.1, semantic_sim=0.5),
        PairCandidate("q1", "a4", "b4", bootstrap_margin=0.01, semantic_sim=0.5),
    ]


def _make_complete_graph_pairs():
    # a fully-connected pair universe (all C(4,2) combos) among 4 docs in one query,
    # since schedule_random_cycles expects complete per-query coverage to build cycles
    import itertools
    docs = ["d0", "d1", "d2", "d3"]
    return [
        PairCandidate("q1", a, b, bootstrap_margin=0.5, semantic_sim=0.5)
        for a, b in itertools.combinations(docs, 2)
    ]


def test_random_cycles_truncates_to_budget():
    pairs = _make_complete_graph_pairs()
    out = schedule_random_cycles(pairs, budget=3, seed=0)
    assert len(out) <= 3
    assert set(p.doc_a_id for p in out) <= {p.doc_a_id for p in pairs}


def test_random_cycles_terminates_with_incomplete_pair_graph():
    # regression test: a query with more docs than available pair edges (sparse,
    # non-complete graph) must not hang -- q_budget has to be capped at the number
    # of pairs actually available for that query, not just at n docs.
    pairs = [
        PairCandidate("q1", "d0", "d1", bootstrap_margin=0.5, semantic_sim=0.5),
        PairCandidate("q1", "d2", "d3", bootstrap_margin=0.5, semantic_sim=0.5),
    ]
    out = schedule_random_cycles(pairs, budget=10, seed=0)
    assert len(out) <= len(pairs)


def test_compositional_orders_easy_first():
    pairs = _make_pairs()
    out = schedule_compositional(pairs, budget=5, seed=0)
    margins = [p.bootstrap_margin for p in out]
    assert margins == sorted(margins, reverse=True)
    assert margins[0] == 1.0
    assert margins[-1] == 0.01


def test_anti_compositional_orders_hard_first():
    pairs = _make_pairs()
    out = schedule_anti_compositional(pairs, budget=5, seed=0)
    margins = [p.bootstrap_margin for p in out]
    assert margins == sorted(margins)
    assert margins[0] == 0.01
    assert margins[-1] == 1.0


def test_compositional_and_anti_are_reverses_of_each_other():
    pairs = _make_pairs()
    easy_first = schedule_compositional(pairs, budget=5, seed=0)
    hard_first = schedule_anti_compositional(pairs, budget=5, seed=0)
    assert [p.doc_a_id for p in easy_first] == [p.doc_a_id for p in reversed(hard_first)]


def test_compositional_respects_budget_and_is_deterministic():
    pairs = _make_pairs()
    out1 = schedule_compositional(pairs, budget=3, seed=1)
    out2 = schedule_compositional(pairs, budget=3, seed=1)
    assert len(out1) == 3
    assert [p.doc_a_id for p in out1] == [p.doc_a_id for p in out2]


def _make_many_pairs(n=60):
    # enough pairs (>> N_DIFFICULTY_BINS) that each difficulty bucket holds several pairs,
    # so within-bucket shuffling has room to actually produce different orderings per seed.
    import random as _random
    rng = _random.Random(123)
    return [
        PairCandidate("q1", f"a{i}", f"b{i}", bootstrap_margin=rng.random(), semantic_sim=rng.random())
        for i in range(n)
    ]


def test_compositional_same_seed_is_reproducible():
    pairs = _make_many_pairs()
    out1 = schedule_compositional(pairs, budget=60, seed=7)
    out2 = schedule_compositional(pairs, budget=60, seed=7)
    assert [(p.doc_a_id, p.doc_b_id) for p in out1] == [(p.doc_a_id, p.doc_b_id) for p in out2]


def test_compositional_different_seeds_produce_different_order():
    # regression test: before bucketing, a strict full sort by a continuous difficulty
    # score had no real ties, so shuffling before sorting was a no-op and every seed
    # produced the identical schedule. Bucketing must give different seeds real variance.
    pairs = _make_many_pairs()
    out1 = schedule_compositional(pairs, budget=60, seed=1)
    out2 = schedule_compositional(pairs, budget=60, seed=2)
    order1 = [(p.doc_a_id, p.doc_b_id) for p in out1]
    order2 = [(p.doc_a_id, p.doc_b_id) for p in out2]
    assert order1 != order2


def test_random_cycles_reproducible_across_separate_processes():
    # regression test: schedule_random_cycles used to build its per-query doc list via a
    # set comprehension, whose iteration order is randomized per-process by Python's
    # string hash randomization (PYTHONHASHSEED). That meant two separate process
    # invocations with the identical seed could still produce different schedules --
    # breaking judge-response cache reuse across script runs. Spawn genuinely separate
    # interpreters (a single pytest process has one fixed hash seed for its whole run,
    # so this can't be caught by an in-process comparison) and confirm they agree.
    import subprocess
    import sys

    script = (
        "import itertools\n"
        "from src.curricula import PairCandidate, schedule_random_cycles\n"
        "docs = [f'd{i}' for i in range(8)]\n"
        "pairs = [PairCandidate('q1', a, b, bootstrap_margin=0.5, semantic_sim=0.5) "
        "for a, b in itertools.combinations(docs, 2)]\n"
        "out = schedule_random_cycles(pairs, budget=10, seed=42)\n"
        "print([(p.doc_a_id, p.doc_b_id) for p in out])\n"
    )
    results = set()
    for _ in range(3):
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True,
        )
        results.add(proc.stdout.strip())
    assert len(results) == 1, f"schedule differed across separate processes: {results}"


def test_compositional_bucket_macro_order_still_easy_first():
    # macro-level curriculum shape must survive bucketing: the first chunk of the
    # schedule should have a meaningfully higher average difficulty score than the last
    # chunk, even though exact pair-by-pair order within a bucket is now seed-shuffled.
    from src.curricula import _difficulty_score
    pairs = _make_many_pairs()
    out = schedule_compositional(pairs, budget=60, seed=0)
    scores = [_difficulty_score(p) for p in out]
    first_chunk = scores[:10]
    last_chunk = scores[-10:]
    assert sum(first_chunk) / len(first_chunk) > sum(last_chunk) / len(last_chunk)
