"""Tests for evaluation metrics."""

from evaluation.metrics import (
    exact_match, token_bleu, codebleu,
    retrieval_precision, hallucination_rate, compute_all_metrics,
)


class TestMetrics:
    def test_exact_match(self):
        assert exact_match("hello", "hello") == 1.0
        assert exact_match("hello", "world") == 0.0

    def test_bleu_identical(self):
        score = token_bleu("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.9

    def test_codebleu(self):
        pred = "class Foo:\n    def bar(self):\n        return 1"
        ref = "class Foo:\n    def bar(self):\n        return 2"
        assert codebleu(pred, ref) > 0.3

    def test_retrieval_precision(self):
        assert retrieval_precision(["a.py", "b.py"], ["a.py"]) == 0.5

    def test_compute_all_metrics(self):
        metrics = compute_all_metrics("answer", "answer", latency_ms=100, tokens=10)
        assert "bleu" in metrics
        assert "latency_ms" in metrics
