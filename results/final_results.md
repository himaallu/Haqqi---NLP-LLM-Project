# Final results (human mode)

From the Colab run of `notebooks/Haqqi_MAAI902.ipynb` on an A100 (28 Oct 2026); the executed notebook with all outputs is `notebooks/Haqqi_MAAI902_results.ipynb`.
Test set: the 91 RQ1 test questions (48 Arabic, 43 English), labelled blind by the author (`data/labelled/test_labels.csv`). Training: the other 360 silver-labelled questions.

## Labels
Silver (LLM) labels vs the author's labels on the 91 test questions: agreement 0.88, Cohen's kappa 0.82.

## RQ1: problem-type classification (macro-F1 / accuracy)

| Model | All | Arabic | English | Accuracy (all) |
|---|---|---|---|---|
| BGE-M3 + LR | **0.755** | **0.704** | **0.793** | **0.791** |
| Qwen3-8B (zero-shot) | 0.662 | 0.573 | 0.736 | 0.725 |
| Jais 2 8B Chat (zero-shot) | 0.490 | 0.547 | 0.398 | 0.429 |
| TF-IDF + LR | 0.420 | 0.402 | 0.256 | 0.681 |

Qwen3-8B gave 1 unreadable reply out of 91 after 3 tries; it is counted as wrong. Small classes in the test set: gratuity 1, passport_withheld 2, recruitment_fees 4.

## RQ2: fact extraction (precision / recall)

| Fact | Gold | Rules P / R | LLM (Qwen3-8B) P / R |
|---|---|---|---|
| amount | 14 | **0.917** / 0.786 | 0.650 / **0.929** |
| date_mentioned | 13 | 0.571 / 0.308 | **0.636** / **0.538** |
| duration | 14 | 0.474 / 0.643 | **0.542** / **0.929** |
| employer | 0 | 0 predicted | 16 predicted, all wrong |

No test question names a real employer; the LLM's 16 employers are generic phrases (e.g. "a contracting company").

## RQ3: article retrieval and answers

Retrieval, the 42 test questions whose problem type has articles in the map (27 Arabic, 15 English; "other" is not scored):

| Hops | Language | recall@3 | hit@3 |
|---|---|---|---|
| one hop | ar | 0.167 | 0.481 |
| one hop | en | 0.133 | 0.267 |
| one hop | all | 0.155 | 0.405 |
| two hops | ar | 0.469 | 0.963 |
| two hops | en | 0.489 | 0.933 |
| two hops | all | **0.476** | **0.952** |

Answers by Qwen3-8B, 30 test questions (15 per language), three conditions:

| Condition | Lang | Similarity to lawyer answer | Cites any article | Cites a mapped article* | Same language |
|---|---|---|---|---|---|
| model alone | ar | 0.649 | 0.733 | 0.000 | 1.0 |
| model alone | en | 0.765 | 0.800 | 0.000 | 1.0 |
| + Q&As | ar | 0.678 | 0.600 | 0.000 | 1.0 |
| + Q&As | en | 0.782 | 0.667 | 0.000 | 1.0 |
| + Q&As + articles | ar | 0.657 | 0.733 | 0.429 | 1.0 |
| + Q&As + articles | en | 0.764 | 0.933 | 0.833 | 1.0 |

\* Only questions whose problem type has articles in the map: 7 Arabic, 6 English.

## Synthetic check (17 made-up cases: 13 English, 4 Arabic)

| Check | Method | All | Arabic | English |
|---|---|---|---|---|
| problem type accuracy | Qwen3-8B | 0.941 | 0.750 | 1.000 |
| problem type accuracy | BGE-M3 + LR | 0.882 | 0.750 | 0.923 |
| problem type accuracy | Jais 2 8B Chat | 0.824 | 1.000 | 0.769 |
| problem type accuracy | TF-IDF + LR | 0.412 | 0.750 | 0.308 |
| key articles hit@3 | one hop / two hops | 0.625 / 1.000 | 0.500 / 1.000 | 0.667 / 1.000 |
| key articles recall@3 | one hop / two hops | 0.458 / 0.823 | 0.333 / 0.917 | 0.500 / 0.792 |

## Figures
`results/figures/`: fig1_data, fig2_labels, fig3_rq1_human, fig4_rq2_human, fig5_rq3_recall_human, fig6_rq3_answers_human.
