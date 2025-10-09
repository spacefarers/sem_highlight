"""Baseline keyword highlighting experiment.

This script loads the Abstract Keyphrases dataset from Hugging Face,
fits a TF-IDF model on the training abstracts, and evaluates a simple
term-ranking baseline on the test split. The baseline selects the top
``k`` n-grams for each document, where ``k`` equals the number of gold
keywords in that document. Performance is reported using micro-averaged
precision, recall, and F1 score computed over exact token matches, where
predicted n-grams and gold keyphrases are both normalized and split into
tokens.

Run with ``python src/baseline.py``.
"""
from __future__ import annotations

import re
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import snapshot_download

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_KP20K_SAMPLE_SIZE = 10_000


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def load_abstract_keyphrases() -> DatasetDict:
    """Load the Abstract Keyphrases dataset from the Hugging Face Hub."""

    dataset = load_dataset("Adapting/abstract-keyphrases")
    return dataset


def determine_kp20k_sample_size() -> int | None:
    """Read the optional KP20k sample size from ``KP20K_TRAIN_DOCS``."""

    raw_value = os.getenv("KP20K_TRAIN_DOCS")
    if raw_value is None:
        return DEFAULT_KP20K_SAMPLE_SIZE

    raw_value = raw_value.strip().lower()
    if raw_value in {"", "none"}:
        return None

    try:
        parsed = int(raw_value)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(
            "KP20K_TRAIN_DOCS must be an integer, 'none', or empty"
        ) from exc

    if parsed < 0:
        raise ValueError("KP20K_TRAIN_DOCS must be non-negative")
    return parsed


def load_kp20k_train_texts(max_documents: int | None) -> List[str]:
    """Load a sample of KP20k abstracts for additional training data."""

    if max_documents == 0:
        return []
    repo_dir = snapshot_download(
        "midas/kp20k",
        repo_type="dataset",
        allow_patterns=["train.jsonl"],
    )
    train_path = Path(repo_dir) / "train.jsonl"
    if not train_path.exists():  # pragma: no cover - defensive branch
        raise FileNotFoundError(
            "Expected train.jsonl in KP20k dataset download"
        )

    texts: List[str] = []
    with train_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            tokens = record.get("document", [])
            if not tokens:
                continue
            text = " ".join(tokens)
            if text:
                texts.append(text)
            if max_documents is not None and len(texts) >= max_documents:
                break
    return texts


def parse_keywords(raw_keywords: str) -> List[str]:
    """Split the semicolon-delimited keyword string into normalized phrases."""

    if raw_keywords is None:
        return []
    keywords = [kw.strip().lower() for kw in raw_keywords.split(";")]
    return [kw for kw in keywords if kw]


def select_top_phrases(
    tfidf_vectorizer: TfidfVectorizer,
    document: str,
    k: int,
) -> List[str]:
    """Select the ``k`` highest-scoring n-grams for ``document``.

    Parameters
    ----------
    tfidf_vectorizer:
        The fitted ``TfidfVectorizer`` instance.
    document:
        Input text to transform.
    k:
        Number of n-grams to select. If ``k`` is zero, an empty list is
        returned.
    """

    if k <= 0:
        return []

    row = tfidf_vectorizer.transform([document])
    row_coo = row.tocoo()
    if row_coo.nnz == 0:
        return []

    scores = row_coo.data
    indices = row_coo.col

    top_indices = np.argsort(scores)[::-1][:k]
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    return feature_names[indices[top_indices]].tolist()


def phrases_to_token_set(phrases: Sequence[str]) -> set[str]:
    """Tokenize a sequence of phrases using the evaluation token pattern."""

    tokens: List[str] = []
    for phrase in phrases:
        tokens.extend(TOKEN_PATTERN.findall(phrase.lower()))
    return set(tokens)


def evaluate_predictions(
    predictions: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
) -> EvaluationResult:
    """Compute micro-averaged precision, recall, and F1 score."""

    tp = fp = fn = 0
    for pred_phrases, ref_phrases in zip(predictions, references):
        pred_tokens = phrases_to_token_set(pred_phrases)
        ref_tokens = phrases_to_token_set(ref_phrases)
        tp += len(pred_tokens & ref_tokens)
        fp += len(pred_tokens - ref_tokens)
        fn += len(ref_tokens - pred_tokens)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return EvaluationResult(precision, recall, f1, tp, fp, fn)


def build_training_corpus(dataset: DatasetDict) -> Tuple[List[str], Dict[str, int]]:
    """Assemble training texts from the available corpora."""

    texts: List[str] = []
    counts: Dict[str, int] = {}

    train_texts: List[str] = list(dataset["train"]["Abstract"])
    texts.extend(train_texts)
    counts["abstract_keyphrases_train"] = len(train_texts)

    val_texts: List[str] = list(dataset["validation"]["Abstract"])
    texts.extend(val_texts)
    counts["abstract_keyphrases_validation"] = len(val_texts)

    kp20k_sample_size = determine_kp20k_sample_size()
    kp20k_texts = load_kp20k_train_texts(kp20k_sample_size)
    texts.extend(kp20k_texts)
    counts["kp20k_train_sample"] = len(kp20k_texts)

    return texts, counts


def run_baseline() -> Tuple[EvaluationResult, int, int, Dict[str, int]]:
    """Train the baseline and evaluate it on the test split."""

    dataset = load_abstract_keyphrases()
    training_texts, corpus_counts = build_training_corpus(dataset)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200_000,
    )
    vectorizer.fit(training_texts)

    predictions: List[List[str]] = []
    references: List[List[str]] = []

    for example in dataset["test"]:
        document = example["Abstract"]
        gold_phrases = parse_keywords(example["Keywords"])
        k = len(gold_phrases)
        predicted_phrases = select_top_phrases(vectorizer, document, k)

        predictions.append(predicted_phrases)
        references.append(gold_phrases)

    metrics = evaluate_predictions(predictions, references)
    vocab_size = len(vectorizer.get_feature_names_out())
    doc_count = len(training_texts)

    return metrics, vocab_size, doc_count, corpus_counts


def main() -> None:
    metrics, vocab_size, doc_count, corpus_counts = run_baseline()
    print("Baseline results on Abstract Keyphrases test split")
    print(f"TF-IDF vocabulary size: {vocab_size}")
    print(f"Training documents: {doc_count}")
    print("Training corpus breakdown:")
    for name, count in corpus_counts.items():
        print(f"  {name}: {count}")
    print(
        "Precision: "
        f"{metrics.precision:.3f}, Recall: {metrics.recall:.3f}, F1: {metrics.f1:.3f}"
    )
    print(
        "Token counts -> TP: {tp}, FP: {fp}, FN: {fn}".format(
            tp=metrics.true_positives,
            fp=metrics.false_positives,
            fn=metrics.false_negatives,
        )
    )


if __name__ == "__main__":
    main()
