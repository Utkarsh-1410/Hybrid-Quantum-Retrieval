import pytest

from app.evaluation.metrics import evaluate_run
from app.evaluation.statistics import paired_bootstrap


def test_evaluate_run_computes_standard_ir_metrics() -> None:
    qrels = {"q1": {"d1": 1, "d2": 1}}
    run = {"q1": {"d1": 3.0, "noise": 2.0, "d2": 1.0}}

    metrics = evaluate_run(qrels, run, cutoffs=(2,))

    assert metrics["map"] == pytest.approx(0.833333)
    assert metrics["mrr@2"] == pytest.approx(1.0)
    assert metrics["recall@2"] == pytest.approx(0.5)
    assert metrics["ndcg@2"] == pytest.approx(0.613147)


def test_paired_bootstrap_reports_positive_delta() -> None:
    baseline = {
        "q1": {"map": 0.2},
        "q2": {"map": 0.4},
        "q3": {"map": 0.1},
    }
    treatment = {
        "q1": {"map": 0.4},
        "q2": {"map": 0.5},
        "q3": {"map": 0.3},
    }

    result = paired_bootstrap(
        baseline,
        treatment,
        iterations=500,
        seed=7,
    )

    assert result["map"]["mean_delta"] == pytest.approx(0.166667)
    assert result["map"]["ci_95_low"] > 0
