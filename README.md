# Semantic Highlighting Project

This repository explores models for automatically highlighting semantically important spans in dense web content. The long-term goal is a lightweight browser extension that keeps full text intact while visually emphasizing salient information.

## Getting started

1. Create a Python 3.11 environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Set the number of KP20k training documents to sample. The
   default is 10,000 abstracts; you can lower this for quicker local runs or
   set it to `0` to skip the auxiliary corpus entirely:
   ```bash
   export KP20K_TRAIN_DOCS=5000  # adjust as desired
   ```
4. Run the baseline experiment:
   ```bash
   python src/baseline.py
   ```

The baseline downloads the [Abstract Keyphrases dataset](https://huggingface.co/datasets/Adapting/abstract-keyphrases) and a
sample of the [KP20k scientific abstracts](https://huggingface.co/datasets/midas/kp20k). It fits a TF-IDF ranker on the combined
corpus (Abstract Keyphrases train + validation + KP20k sample) and reports micro-averaged precision, recall, and F1 on the
Abstract Keyphrases test split using token-level overlap. Expect the first run to download ~1.4 GB for the KP20k JSONL file.

## Reports

* `reports/milestone2.md` – Milestone 2 write-up covering data sources, preprocessing plan, evaluation metrics, and baseline results.
