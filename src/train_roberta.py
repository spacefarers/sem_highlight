"""RoBERTa token-classification baseline for OpenKP keyphrase extraction.

This script trains a RoBERTa model to tag tokens with BIO labels and
extracts ranked keyphrase spans from the predictions. Run with
``python src/train_roberta.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
from torch.nn import functional as F
from transformers import (
    DataCollatorForTokenClassification,
    RobertaForTokenClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)

from baseline import (
    EvaluationResult,
    evaluate_predictions,
    normalize_phrases,
    determine_max_documents,
)

LABEL_NAMES = ["O", "B-KEY", "I-KEY"]
LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(LABEL_NAMES)}
ID2LABEL: Dict[int, str] = {i: label for label, i in LABEL2ID.items()}
MAX_LENGTH = 512
DEFAULT_OPENKP_TRAIN_DOCS = 20_000


def maybe_limit_records(records: List[Dict], max_docs: int | None) -> List[Dict]:
    """Optionally truncate a list of dict records."""

    if max_docs is None or max_docs >= len(records):
        return records
    return records[:max_docs]


def load_openkp_records() -> Dict[str, List[Dict]]:
    """Load OpenKP JSONL files using the same snapshot approach as the baseline."""

    repo_dir = Path(
        snapshot_download(
            "midas/openkp",
            repo_type="dataset",
            allow_patterns=["train.jsonl", "valid.jsonl", "test.jsonl"],
        )
    )

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

    datasets: Dict[str, List[Dict]] = {}
    for split_name, filename in split_files.items():
        path = repo_dir / filename
        records: List[Dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                document = record.get("document") or []
                if not document:
                    continue
                doc_bio_tags = record.get("doc_bio_tags") or []
                keyphrases = normalize_phrases(record.get("extractive_keyphrases") or [])
                records.append(
                    {
                        "document": document,
                        "doc_bio_tags": doc_bio_tags,
                        "extractive_keyphrases": keyphrases,
                    }
                )
                if (
                    max_docs_by_split[split_name] is not None
                    and len(records) >= max_docs_by_split[split_name]
                ):
                    break

        datasets[split_name] = records

    return datasets


def tokenize_and_align_labels(
    examples: Dict[str, Sequence],
    tokenizer: RobertaTokenizerFast,
) -> Dict[str, List[List[int]]]:
    """Tokenize OpenKP documents and align BIO tags to subwords."""

    tokenized = tokenizer(
        examples["document"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    all_labels: List[List[int]] = []
    all_word_ids: List[List[int]] = []
    for batch_idx, word_labels in enumerate(examples["doc_bio_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_idx)
        aligned_labels: List[int] = []
        aligned_word_ids: List[int] = []

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
                aligned_word_ids.append(-1)
            elif word_idx != previous_word_idx:
                raw_label = word_labels[word_idx]
                label_id: int
                if isinstance(raw_label, str):
                    upper = raw_label.upper()
                    if upper in {"B", "B-KEY"}:
                        label_id = LABEL2ID["B-KEY"]
                    elif upper in {"I", "I-KEY"}:
                        label_id = LABEL2ID["I-KEY"]
                    else:
                        label_id = LABEL2ID["O"]
                else:
                    try:
                        numeric = int(raw_label)
                    except Exception:
                        numeric = 0
                    label_id = int(numeric)

                aligned_labels.append(label_id)
                aligned_word_ids.append(word_idx)
            else:
                aligned_labels.append(-100)
                aligned_word_ids.append(-1)
            previous_word_idx = word_idx

        all_labels.append(aligned_labels)
        all_word_ids.append(aligned_word_ids)

    tokenized["labels"] = all_labels
    tokenized["word_ids"] = all_word_ids
    return tokenized


def prepare_datasets(
    tokenizer: RobertaTokenizerFast,
) -> Tuple[DatasetDict, List[List[int]], List[List[str]], List[List[str]]]:
    """Load, optionally slice, and tokenize OpenKP splits using snapshot data."""

    raw_records = load_openkp_records()
    raw_datasets = DatasetDict(
        {split: Dataset.from_list(records) for split, records in raw_records.items()}
    )

    tokenized = raw_datasets.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        desc="Tokenizing OpenKP",
    )

    test_word_ids = tokenized["test"]["word_ids"]
    test_documents = raw_datasets["test"]["document"]
    test_keyphrases = raw_datasets["test"]["extractive_keyphrases"]

    columns_to_keep = {"input_ids", "attention_mask", "labels"}
    processed = DatasetDict()
    for split_name, split_ds in tokenized.items():
        keep_cols = [col for col in split_ds.column_names if col in columns_to_keep]
        processed[split_name] = split_ds.remove_columns(
            [col for col in split_ds.column_names if col not in columns_to_keep]
        ).with_format("python", columns=keep_cols)

    return processed, test_word_ids, test_documents, test_keyphrases


def build_trainer(
    tokenized_datasets: DatasetDict,
    tokenizer: RobertaTokenizerFast,
) -> Trainer:
    """Configure the Trainer for token classification."""

    model = RobertaForTokenClassification.from_pretrained(
        "roberta-base",
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir="roberta_openkp_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets.get("validation", None)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


def extract_ranked_phrases(
    logits: np.ndarray,
    word_id_sequences: List[List[int]],
    documents: List[List[str]],
) -> List[List[str]]:
    """Convert token-level predictions into ranked, deduplicated phrases."""

    all_phrases: List[List[str]] = []
    for doc_logits, word_ids, words in zip(logits, word_id_sequences, documents):
        probs = F.softmax(torch.tensor(doc_logits), dim=-1).cpu().numpy()

        word_labels: List[int] = []
        word_scores: List[float] = []
        previous_word_idx = -1
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx == -1 or word_idx == previous_word_idx:
                continue
            label_id = int(np.argmax(probs[token_idx]))
            label_score = float(probs[token_idx][label_id])
            word_labels.append(label_id)
            word_scores.append(label_score)
            previous_word_idx = word_idx

        phrases_with_scores: Dict[str, float] = {}
        position = 0
        while position < len(word_labels):
            label_id = word_labels[position]
            if label_id == LABEL2ID["B-KEY"]:
                start = position
                scores = [word_scores[position]]
                position += 1
                while (
                    position < len(word_labels)
                    and word_labels[position] == LABEL2ID["I-KEY"]
                ):
                    scores.append(word_scores[position])
                    position += 1
                end = position
                if start < len(words):
                    phrase_tokens = words[start:end]
                    phrase = " ".join(phrase_tokens).strip().lower()
                    if phrase:
                        mean_score = float(np.mean(scores))
                        best_score = phrases_with_scores.get(phrase)
                        if best_score is None or mean_score > best_score:
                            phrases_with_scores[phrase] = mean_score
            else:
                position += 1

        ranked = sorted(
            phrases_with_scores.items(), key=lambda item: item[1], reverse=True
        )
        all_phrases.append([phrase for phrase, _ in ranked])

    return all_phrases


def evaluate_at_k(
    predictions: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
    k: int,
) -> EvaluationResult:
    """Evaluate predictions using the top-k phrases for each example."""

    top_k_predictions = [list(phrases[:k]) for phrases in predictions]
    return evaluate_predictions(top_k_predictions, references)


def run_inference(
    model_dir: str,
    documents: List[List[str]],
    max_length: int = MAX_LENGTH,
) -> List[List[str]]:
    """Load a saved model and return ranked keyphrases for tokenized documents."""

    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir, add_prefix_space=True)
    model = RobertaForTokenClassification.from_pretrained(model_dir)
    model.eval()

    tokenized = tokenizer(
        documents,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = [tokenized.word_ids(i) for i in range(len(documents))]

    with torch.no_grad():
        outputs = model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
        )

    return extract_ranked_phrases(
        outputs.logits.detach().cpu().numpy(), word_ids, documents
    )


def main() -> None:
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "roberta-base", add_prefix_space=True
    )
    tokenized_datasets, test_word_ids, test_documents, test_keyphrases = (
        prepare_datasets(tokenizer)
    )

    trainer = build_trainer(tokenized_datasets, tokenizer)
    trainer.train()

    prediction_output = trainer.predict(tokenized_datasets["test"])
    predicted_phrases = extract_ranked_phrases(
        prediction_output.predictions, test_word_ids, test_documents
    )

    f1_at_1 = evaluate_at_k(predicted_phrases, test_keyphrases, k=1)
    f1_at_3 = evaluate_at_k(predicted_phrases, test_keyphrases, k=3)
    f1_at_5 = evaluate_at_k(predicted_phrases, test_keyphrases, k=5)
    f1_at_7 = evaluate_at_k(predicted_phrases, test_keyphrases, k=7)

    model_dir = "roberta_openkp_model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("RoBERTa token classification results on OpenKP test split")
    print(f"F1@1: {f1_at_1.f1:.3f} (P={f1_at_1.precision:.3f}, R={f1_at_1.recall:.3f})")
    print(f"F1@3: {f1_at_3.f1:.3f} (P={f1_at_3.precision:.3f}, R={f1_at_3.recall:.3f})")
    print(f"F1@5: {f1_at_5.f1:.3f} (P={f1_at_5.precision:.3f}, R={f1_at_5.recall:.3f})")
    print(f"F1@7: {f1_at_7.f1:.3f} (P={f1_at_7.precision:.3f}, R={f1_at_7.recall:.3f})")
    print(f"Saved model and tokenizer to: {model_dir}")


if __name__ == "__main__":
    main()
