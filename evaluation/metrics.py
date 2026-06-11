"""Evaluation metrics for ArchonAI model comparison."""

from __future__ import annotations

import re
import time
from typing import Any


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if prediction.strip() == reference.strip() else 0.0


def token_bleu(prediction: str, reference: str, n: int = 4) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    scores = []
    for i in range(1, n + 1):
        pred_ngrams = _ngrams(pred_tokens, i)
        ref_ngrams = _ngrams(ref_tokens, i)
        if not pred_ngrams:
            continue
        overlap = sum(min(pred_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in pred_ngrams)
        scores.append(overlap / max(sum(pred_ngrams.values()), 1))
    return sum(scores) / len(scores) if scores else 0.0


def _ngrams(tokens: list[str], n: int) -> dict[tuple, int]:
    counts: dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def codebleu(prediction: str, reference: str) -> float:
    """Simplified CodeBLEU using token overlap and AST keyword matching."""
    bleu = token_bleu(prediction, reference)
    pred_keywords = set(re.findall(r"\b(class|def|import|from|return|async|await)\b", prediction))
    ref_keywords = set(re.findall(r"\b(class|def|import|from|return|async|await)\b", reference))
    keyword_overlap = len(pred_keywords & ref_keywords) / max(len(ref_keywords), 1)
    return 0.6 * bleu + 0.4 * keyword_overlap


def retrieval_precision(retrieved: list[str], relevant: list[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if any(rel in r for rel in relevant))
    return hits / len(retrieved)


def retrieval_recall(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for rel in relevant if any(rel in r for r in retrieved))
    return hits / len(relevant)


def hallucination_rate(response: str, sources: list[str]) -> float:
    """Estimate hallucination rate based on uncited claims."""
    code_blocks = re.findall(r"```[\s\S]*?```", response)
    if not code_blocks:
        return 0.0
    source_text = " ".join(sources).lower()
    uncited = sum(
        1 for block in code_blocks
        if not any(line.strip()[:30] in source_text for line in block.split("\n") if line.strip())
    )
    return uncited / len(code_blocks)


def pass_at_k(results: list[bool], k: int = 1) -> float:
    if not results:
        return 0.0
    successes = sum(results[:k])
    return 1.0 if successes > 0 else 0.0


def compute_all_metrics(
    prediction: str,
    reference: str,
    retrieved: list[str] | None = None,
    relevant: list[str] | None = None,
    latency_ms: float = 0.0,
    tokens: int = 0,
) -> dict[str, float]:
    retrieved = retrieved or []
    relevant = relevant or []
    return {
        "exact_match": exact_match(prediction, reference),
        "bleu": token_bleu(prediction, reference),
        "codebleu": codebleu(prediction, reference),
        "retrieval_precision": retrieval_precision(retrieved, relevant),
        "retrieval_recall": retrieval_recall(retrieved, relevant),
        "hallucination_rate": hallucination_rate(prediction, retrieved),
        "pass_at_1": pass_at_k([exact_match(prediction, reference) == 1.0], 1),
        "latency_ms": latency_ms,
        "token_usage": tokens,
    }
