# Milestone 2: Data, Baseline, and Evaluation Plan

## Project overview and pivot rationale
The project now targets a **semantic highlight assistant for long-form web content**. Instead of restricting the scope to email clients, the system will ingest any high-density text page—news articles, technical blog posts, product documentation, or research papers—and surface the most semantically important spans directly within the page. The long-term deliverable is a lightweight browser extension that can run locally, injecting color-coded emphasis (e.g., highlight overlays) on the most salient tokens or phrases without altering the original prose. This pivot maintains the emphasis on rapid comprehension while broadening the applicability to any text-heavy web experience.

## Data resources
To make progress quickly while still covering a variety of writing styles, I am curating a blend of public datasets and self-collected material.

1. **Abstract Keyphrases dataset** ([Hugging Face](https://huggingface.co/datasets/Adapting/abstract-keyphrases)) – 90 annotated scientific abstracts (50 train / 20 validation / 20 test) with semicolon-separated gold keyphrases. Although small, it provides high-quality expert annotations and a controlled scientific writing style. Downloaded programmatically in the baseline script via the `datasets` Python library.
2. **KP20k scientific abstracts** ([Hugging Face](https://huggingface.co/datasets/midas/kp20k)) – 527K research paper abstracts paired with author-assigned keyphrases. The baseline now samples 10K training documents from the `train.jsonl` split (configurable via `KP20K_TRAIN_DOCS`) to expand beyond the 50-document Abstract Keyphrases train set while keeping runtime reasonable. Downloaded automatically with `huggingface_hub.snapshot_download`.
3. **Self-curated web pages** – For the browser-centered pivot, I am compiling a small evaluation set (~100 documents) of blog posts, engineering changelogs, and policy documents. Each document will be annotated using a combination of personal labeling and LLM-assisted bootstrapping to mark the top ~10 highlights per page. These annotations will serve as the final integration test set for the extension experience.

## Preprocessing pipeline
Preprocessing focuses on normalizing text to support both statistical baselines and future neural methods.

* **Normalization** – Lowercasing, HTML tag stripping (for scraped pages), Unicode normalization, and whitespace collapse. For structured datasets (Abstract Keyphrases, KP20k) this is minimal because the text arrives clean.
* **Segmentation** – Sentence splitting using spaCy to provide optional context windows during highlighting and to support later heuristics (e.g., ensuring coverage across the document).
* **Candidate generation** – Tokenization using a lightweight regular-expression tokenizer, plus optional n-gram assembly (1–3 grams). Candidate spans shorter than three characters or consisting entirely of stop words are filtered out. In the browser, DOM-aware candidate extraction will map tokens back to offsets for rendering highlights.
* **Feature preparation** – For classical baselines, compute TF-IDF statistics and simple positional features; for future neural models, build token-ID vocabularies and document embeddings (e.g., MiniLM). All steps are scripted in Python notebooks to enable reproducibility.

## Evaluation protocol
The core metric is **micro-averaged F1 on highlighted tokens**. Both gold keyphrases and model predictions are normalized (lowercase, alphanumeric tokens) before computing overlap. Token-level scoring is appropriate because the end-user experience involves highlighting contiguous tokens; partial phrase matches still give value, which phrase-level exact match would ignore. Secondary metrics include coverage (fraction of sentences with at least one highlight) and redundancy (duplicate token highlights). Human evaluation is planned for the browser prototype to ensure the highlights feel meaningful and not distracting.

## Baseline system
The initial baseline is an unsupervised TF-IDF ranker implemented in `src/baseline.py`. Steps:

1. Assemble a training corpus consisting of Abstract Keyphrases train + validation splits (70 documents) and a configurable sample from KP20k (10K abstracts by default via `KP20K_TRAIN_DOCS`).
2. Fit a `TfidfVectorizer` with unigram and bigram features, English stop-word removal, and a 200K feature cap to keep the vocabulary tractable while scaling to the larger corpus.
3. For each test document, select the top *k* n-grams, where *k* equals the number of gold keyphrases for that document, convert predictions and references to token sets, and compute precision, recall, and F1.

Running `python src/baseline.py` (with the default 10K KP20k sample) yields:

```
Baseline results on Abstract Keyphrases test split
TF-IDF vocabulary size: 200000
Training documents: 10070
Training corpus breakdown:
  abstract_keyphrases_train: 50
  abstract_keyphrases_validation: 20
  kp20k_train_sample: 10000
Precision: 0.635, Recall: 0.217, F1: 0.324
Token counts -> TP: 40, FP: 23, FN: 144
```

The added KP20k coverage boosts precision to 0.635 (+0.103) and recall to 0.217 (+0.038), producing a relative F1 improvement of ~21%. The vectorizer now encounters a wider range of scientific terminology, which reduces false positives while uncovering a few more gold tokens. Recall remains modest because TF-IDF still struggles with multiword phrases and common-but-important terminology, underscoring the need for context-aware models.

## Feasibility and next steps
The pipeline now reliably downloads data, executes the baseline, and produces measurable metrics, demonstrating project feasibility. Upcoming milestones will focus on: (1) expanding the training corpus with deeper KP20k coverage, KPTimes news articles, and the self-curated web pages, (2) experimenting with embedding-based unsupervised methods such as TextRank or Maximal Marginal Relevance using sentence transformers, (3) integrating a lightweight neural keyphrase extractor fine-tuned on the combined datasets, and (4) building a DOM-aware highlighting prototype to validate the UX on live pages. The milestone baseline and evaluation harness provide a scaffold to quantify progress as more sophisticated models are introduced.
