# Multilingual Health Question Answering in Low-Resource African Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nick-Lemy/multilingual-health-qa/blob/main/multilingual_health_qa_starter_notebook.ipynb)

Solution for the Zindi **Multilingual Health Question Answering in Low-Resource
African Languages** challenge (sponsored by ITU/HASH). Given a health question in a
low-resource African language, the model generates a fluent, accurate answer **in the
same language**.

**Languages in the data:** English, Akan, Luganda, Swahili, Amharic
(across Uganda, Ghana, Ethiopia, Kenya).

## Scoring

Leaderboard score = weighted mean of three metrics:

| Metric | Weight |
|--------|--------|
| ROUGE-1 F1 | 0.37 |
| ROUGE-L F1 | 0.37 |
| LLM-as-a-Judge (1–5, normalised) | 0.26 |

Submission columns: `ID`, `TargetRLF1`, `TargetR1F1`, `TargetLLM` — the same predicted
answer is placed in all three target columns.

## Quick start (Colab — recommended)

1. Click the **Open In Colab** badge above.
2. Runtime → **Change runtime type → GPU** (needed for the LLM / fine-tuning sections).
3. **Run all cells.** Data loads automatically — it uses local `./data` if present,
   otherwise downloads the CSVs from this repo's raw URLs. **No Google Drive or account
   sign-in required.**

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook multilingual_health_qa_starter_notebook.ipynb
```

The lightweight sections (EDA, TF-IDF baseline, ROUGE, experiment tracking) run on CPU.
Fine-tuning needs a GPU — use Colab for that.

## Pipeline (notebook sections)

1. Install & imports
2. Account-free data loading (local → GitHub-raw fallback)
3. Load & preview
4. **EDA** — language balance, length distributions, data-quality checks
5. Text cleaning
7. ROUGE evaluation utilities · **7b — experiment tracking**
8. Baseline 1 — TF-IDF nearest-neighbour retrieval (char n-grams)
9. Baseline 2 — multilingual LLM (mT5)
10. Submission file creation
11. Baseline comparison
12. Fine-tuning mT5 on the training data
13. **Experiment roadmap** (≥10 experiments) + review

## Experiment tracking

Every meaningful run is logged to [`data/experiments_log.csv`](data/experiments_log.csv)
via `log_experiment(name, config, val_rouge1, val_rougeL, leaderboard, notes)`.
`show_experiments()` ranks them; `plot_experiment_progression()` charts progress.
`val_proxy = 0.37·ROUGE-1 + 0.37·ROUGE-L` is an offline proxy for the official score.

## Reproducibility

- All randomness is seeded (`SEED = 42`).
- Runs end-to-end on Colab with only the dependencies in `requirements.txt`.
- mT5 is trained without fp16 (it produces NaN losses under fp16); bf16 is used only
  when the GPU supports it, otherwise full fp32.

## Notes

The model base is **mT5** (a text-to-text model suited to same-language generation).
NLLB is a *translation* model and is the wrong base for this task — kept only as a
candidate experiment, never the default.
