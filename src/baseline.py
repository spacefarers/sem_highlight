"""Baseline keyword highlighting experiment.

This script loads the OpenKP dataset from the Hugging Face Hub,
fits a TF-IDF model on the training passages, and evaluates a simple
term-ranking baseline on the test split. The baseline selects the top
``k`` n-grams for each document, where ``k`` equals the number of gold
extractive keyphrases in that document. Performance is reported using
micro-averaged precision, recall, and F1 score computed over exact token
matches after normalizing both predictions and references.

Run with ``python src/baseline.py``.
"""
from __future__ import annotations

import json
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_OPENKP_TRAIN_DOCS = 20_000


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class OpenKPExample:
    """Single OpenKP record with joined text and gold phrases."""

    text: str
    keyphrases: List[str]


def determine_max_documents(env_var: str, default: int | None) -> int | None:
    """Parse an optional environment variable controlling dataset size."""

    raw_value = os.getenv(env_var)
    if raw_value is None:
        return default

    raw_value = raw_value.strip().lower()
    if raw_value in {"", "none"}:
        return None

    try:
        parsed = int(raw_value)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(
            f"{env_var} must be an integer, 'none', or empty"
        ) from exc

    if parsed < 0:
        raise ValueError(f"{env_var} must be non-negative")
    return parsed


def load_openkp_dataset() -> Dict[str, List[OpenKPExample]]:
    """Download the OpenKP dataset and return splits as in-memory examples."""

    repo_dir = Path(
        snapshot_download(
            "midas/openkp",
            repo_type="dataset",
            allow_patterns=["train.jsonl", "valid.jsonl", "test.jsonl"],
        )
    )

    dataset: Dict[str, List[OpenKPExample]] = {}
    split_files = {
        "train": "train.jsonl",
        "validation": "valid.jsonl",
        "test": "test.jsonl",
    }

    max_docs_by_split = {
        "train": determine_max_documents("OPENKP_TRAIN_DOCS", DEFAULT_OPENKP_TRAIN_DOCS),
        "validation": determine_max_documents("OPENKP_VALID_DOCS", None),
        "test": determine_max_documents("OPENKP_TEST_DOCS", None),
    }

    for split_name, filename in split_files.items():
        path = repo_dir / filename
        if not path.exists():  # pragma: no cover - defensive branch
            raise FileNotFoundError(
                f"Expected {filename} in OpenKP dataset download"
            )

        examples: List[OpenKPExample] = []
        max_docs = max_docs_by_split[split_name]
        with path.open("r", encoding="utf-8") as handle:
            iterator = tqdm(
                handle,
                desc=f"Loading OpenKP {split_name}",
                unit="record",
                total=max_docs,
                leave=False,
            )
            for line in iterator:
                record = json.loads(line)
                tokens = record.get("document", [])
                if not tokens:
                    continue
                text = " ".join(tokens)
                raw_phrases = record.get("extractive_keyphrases") or []
                keyphrases = normalize_phrases(raw_phrases)
                examples.append(OpenKPExample(text=text, keyphrases=keyphrases))
                if max_docs is not None and len(examples) >= max_docs:
                    break

        dataset[split_name] = examples

    return dataset


def normalize_phrases(raw_phrases: Sequence[str]) -> List[str]:
    """Normalize a sequence of phrases for evaluation."""

    phrases: List[str] = []
    for phrase in raw_phrases:
        cleaned = phrase.strip().lower()
        if cleaned:
            phrases.append(cleaned)
    return phrases


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


def build_training_corpus(
    dataset: Dict[str, List[OpenKPExample]]
) -> Tuple[List[str], Dict[str, int]]:
    """Assemble training texts from the OpenKP train and validation splits."""

    texts: List[str] = []
    counts: Dict[str, int] = {}

    for split_name in ("train", "validation"):
        split_examples = dataset.get(split_name, [])
        split_texts = [example.text for example in split_examples]
        texts.extend(split_texts)
        counts[f"openkp_{split_name}"] = len(split_texts)

    return texts, counts


def run_baseline() -> Tuple[EvaluationResult, int, int, Dict[str, int]]:
    """Train the baseline and evaluate it on the test split."""

    dataset = load_openkp_dataset()
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

    test_examples = dataset["test"]
    for example in tqdm(
        test_examples,
        desc="Evaluating OpenKP",
        unit="doc",
        total=len(test_examples),
        leave=False,
    ):
        gold_phrases = example.keyphrases
        k = len(gold_phrases)
        predicted_phrases = select_top_phrases(vectorizer, example.text, k)

        predictions.append(predicted_phrases)
        references.append(gold_phrases)

    metrics = evaluate_predictions(predictions, references)
    vocab_size = len(vectorizer.get_feature_names_out())
    doc_count = len(training_texts)

    return metrics, vocab_size, doc_count, corpus_counts


def main() -> None:
    metrics, vocab_size, doc_count, corpus_counts = run_baseline()
    print("Baseline results on OpenKP test split")
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
