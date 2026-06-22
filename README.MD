# Multilingual Health Question Answering in Low-Resource African Languages

A hybrid retrieval system for the [Zindi Multilingual Health QA Challenge](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge). For each test question, it finds the most similar training question within the same language and returns that answer. No GPU fine-tuning is required, so the whole pipeline runs in a few minutes on Colab.

Best submission (E08): **0.5677 public / 0.5587 private** on the Zindi leaderboard.

## How it works

Each test question is matched against the training questions in its own language subset using a blend of two signals:

- **TF-IDF** (lexical overlap) for specific or rare medical vocabulary.
- **Multilingual sentence embeddings** (semantic similarity) for paraphrased questions that share no exact words.

The blend weight is tuned per language, and the answer of the best-matching training question is returned. The submission writes the same predicted answer into all three target columns required by the competition.

## Repository structure

```
.
├── notebook.ipynb        # full pipeline: EDA, experiments, submission
├── README.md
└── data/                 # Train.csv, Test.csv, SampleSubmission.csv (see below)
```

## Running it

The notebook runs end-to-end on Google Colab.

1. Open `notebook.ipynb` in Colab and set the runtime to GPU (Runtime → Change runtime type → GPU).
2. Run the cells from top to bottom.

Data is fetched directly over HTTP from the project repository, so there is nothing to upload. A fixed random seed is set throughout, so reruns reproduce the same results.

## Experiments

Twelve experiments were run and tracked, each writing its own submission file and logging its score:

| ID      | Change                                 | Zindi public |
| ------- | -------------------------------------- | ------------ |
| E01–E04 | TF-IDF and semantic baselines          | 0.45–0.51    |
| E05     | Per-language alpha tuning              | 0.5497       |
| E06     | Exact-match shortcut                   | 0.5497       |
| E07     | LaBSE backbone                         | 0.5522       |
| E08     | Hybrid + character n-grams (best)      | 0.5677       |
| E09–E11 | Fusion and length-calibration variants | 0.52–0.55    |
| E12     | Final model on full corpus             | 0.5567       |

Per-language alpha tuning (E05) gave the largest single gain, and character n-grams (E08) gave the best overall result.

## Requirements

```
sentence-transformers
rouge-score
scikit-learn
pandas
numpy
```

All are installed in the first notebook cell.

## License

Data is used under CC-BY-SA 4.0 per the competition terms.
