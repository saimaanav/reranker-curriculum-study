from src.curricula import (
    PairCandidate,
    schedule_anti_curriculum,
    schedule_compositional_curriculum,
    schedule_difficulty_curriculum,
    schedule_random,
)


def _make_pairs():
    # margins: 10 (easy) .. 0.1 (near-tie); lexical_overlap varies for compositional staging
    return [
        PairCandidate("q1", "a0", "b0", bootstrap_margin=10.0, lexical_overlap=0.9, semantic_sim=0.1),
        PairCandidate("q1", "a1", "b1", bootstrap_margin=7.0, lexical_overlap=0.2, semantic_sim=0.8),
        PairCandidate("q1", "a2", "b2", bootstrap_margin=4.0, lexical_overlap=0.5, semantic_sim=0.5),
        PairCandidate("q1", "a3", "b3", bootstrap_margin=1.0, lexical_overlap=0.5, semantic_sim=0.5),
        PairCandidate("q1", "a4", "b4", bootstrap_margin=0.1, lexical_overlap=0.9, semantic_sim=0.1),
    ]


def test_random_truncates_to_budget():
    pairs = _make_pairs()
    out = schedule_random(pairs, budget=3, seed=0)
    assert len(out) == 3
    assert set(p.doc_a_id for p in out) <= {p.doc_a_id for p in pairs}


def test_random_deterministic_given_seed():
    pairs = _make_pairs()
    out1 = schedule_random(pairs, budget=5, seed=42)
    out2 = schedule_random(pairs, budget=5, seed=42)
    assert [p.doc_a_id for p in out1] == [p.doc_a_id for p in out2]


def test_difficulty_curriculum_orders_easy_first():
    pairs = _make_pairs()
    out = schedule_difficulty_curriculum(pairs, budget=5, seed=0)
    margins = [p.bootstrap_margin for p in out]
    assert margins == sorted(margins, reverse=True)
    assert margins[0] == 10.0
    assert margins[-1] == 0.1


def test_anti_curriculum_orders_hard_first():
    pairs = _make_pairs()
    out = schedule_anti_curriculum(pairs, budget=5, seed=0)
    margins = [p.bootstrap_margin for p in out]
    assert margins == sorted(margins)
    assert margins[0] == 0.1
    assert margins[-1] == 10.0


def test_difficulty_and_anti_are_reverses_of_each_other():
    pairs = _make_pairs()
    easy_first = schedule_difficulty_curriculum(pairs, budget=5, seed=0)
    hard_first = schedule_anti_curriculum(pairs, budget=5, seed=0)
    assert [p.doc_a_id for p in easy_first] == [p.doc_a_id for p in reversed(hard_first)]


def test_compositional_curriculum_respects_budget_and_is_deterministic():
    pairs = _make_pairs()
    out1 = schedule_compositional_curriculum(pairs, budget=3, seed=1)
    out2 = schedule_compositional_curriculum(pairs, budget=3, seed=1)
    assert len(out1) == 3
    assert [p.doc_a_id for p in out1] == [p.doc_a_id for p in out2]


def test_compositional_curriculum_puts_hard_distractor_last():
    pairs = _make_pairs()
    out = schedule_compositional_curriculum(pairs, budget=5, seed=0)
    # a4/b4 has small margin + high lexical overlap -> hard-distractor stage, should be last
    assert out[-1].doc_a_id == "a4"
