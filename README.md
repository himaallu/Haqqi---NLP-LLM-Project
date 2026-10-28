# Haqqi: understanding workers' labour problems

MAAI 902 (NLP & LLMs), Assessment 2. A pipeline that reads a worker's message in Arabic or English, classifies the labour problem, extracts key facts, and retrieves the relevant articles of UAE Federal Decree-Law No. 33 of 2021.

## Submission notebook (Google Colab)
`notebooks/Haqqi_MAAI902.ipynb` is the whole project in one notebook (sections 0-10, code inline). To run it:
1. Upload `haqqi_drive_data.zip` to Google Drive and unzip it so that `MyDrive/haqqi/` holds `data/`, `results/` and `docs/` (everything except the raw HTML).
2. Open the notebook in Colab, choose a GPU runtime (A100 or L4 on Colab Pro), add a Hugging Face read token as the Colab secret `HF_TOKEN` (needed for the gated Jais 2 model) and *Run all*.
3. Cells marked **⚡ GPU cell** run the LLMs (Qwen3-8B, Jais 2 8B) with transformers in bfloat16, or 4-bit on smaller GPUs. Their outputs are cached in `MyDrive/haqqi/results/cache/`, so after a disconnect just run them again. All other cells run on CPU.

The executed run with all outputs (A100, 28 Oct 2026) is `notebooks/Haqqi_MAAI902_results.ipynb`; the numbers are summarised in `results/final_results.md` and the charts are in `results/figures/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

## Today's steps

### 1. Collect question links
Open https://legaladviceme.com/questions, apply the **Employment** area-of-law filter, copy the URL. Do not add `?lang=` (robots.txt disallows it).

The scrapers send a contact address in their user agent so the site owner can reach you; set it first (it is kept out of the code): `export SCRAPER_CONTACT=you@example.com`.

```bash
python src/scrape/collect_links.py --start-url "PASTE_URL" --max-pages 2   # test first
python src/scrape/collect_links.py --start-url "PASTE_URL" --max-pages 70  # then the full run
```
Output: `data/raw/links.csv`. Waits 12 s between pages and resumes if stopped.

### 2. Download the question pages
```bash
python src/scrape/fetch_pages.py --limit 20     # test first
python src/scrape/fetch_pages.py --limit 1000   # leave it running (several hours)
```
Output: `data/raw/html/<id>.html`. Already-saved pages are skipped.

**Arabic data: originals only.** The site's Arabic filter also lists English questions that have an Arabic translation. Those keep an English URL slug and serve English text, while questions written in Arabic have an Arabic slug. `collect_links.py` records this as `slug_lang`, and `fetch_pages.py` skips Arabic-listing links with an English slug, so the Arabic set is native Arabic writing, not translation. Add `--include-translations` to fetch them anyway.

**Scrape results (22 Oct 2026).** 3,414 links collected; 999 pages saved (746 English, 253 Arabic originals, i.e. every Arabic original listed). After cleaning, all 999 are kept and tagged with `law_era`: **866 `current` (742 en, 123 ar, 1 mixed) and 133 `pre2022` (129 ar, 3 en, 1 mixed)**. Arabic posting nearly stops after 2023, so the 123 current Arabic questions are close to all the site has. Filter on `law_era == "current"` when the old law would give wrong answers. The language detected from the text matches `slug_lang` on every row.

### 3. Parse the pages into a CSV
```bash
python src/scrape/parse_pages.py --limit 5   # check the CSV by eye
python src/scrape/parse_pages.py
```
Output: `data/raw/questions.csv`. If it reports empty question text, check the `SELECTORS` at the top of `parse_pages.py`.

### 4. Clean
```bash
python src/preprocess/preprocess.py --in data/raw/questions.csv --out data/clean/questions_clean.csv
```
Removes duplicates, tags each question with `law_era` (`pre2022` or `current`, by year asked; nothing is dropped), masks emails, phone numbers, Emirates IDs and URLs, detects the language (ar / en / mixed), and normalises Arabic.

### 5. Label questions
```bash
python src/label/build_queue.py      # once, and again after saving haqqi_50.csv
streamlit run src/label/label_app.py
```
`build_queue.py` writes `data/labelled/queue.csv`, the order the app follows: all Arabic questions (`current`, then `pre2022`), then 200 random English `current` questions, then your Haqqi cases from `data/labelled/haqqi_50.csv` (needs `id` and `text`). Re-running it keeps the same order for the scraped part and never loses saved labels. Haqqi cases are saved with `source = haqqi`, so they can be kept apart as test data.

**Optional: LLM pre-labelling (training set only).** With `ANTHROPIC_API_KEY` set in your environment:
```bash
python src/label/prelabel.py --dry-run    # see the prompt, no API call
python src/label/prelabel.py --limit 5    # try a few
python src/label/prelabel.py              # the rest (resumes)
```
Claude (`claude-opus-5-5`) suggests labels and facts for the scraped questions, following the same guide, and saves them to `data/labelled/prelabels.csv`. The app then opens those questions pre-filled. Check every field before saving. Your label and the LLM's suggestion are both saved (`llm_problem_type`, `llm_asker`), so you can report how often you changed it. **Haqqi cases are never sent to the LLM or pre-filled**: they are the test set and you label them blind.

**Blind check of the LLM labels.** A fixed random sample of 30 Arabic + 30 English questions, labelled with the LLM's suggestions hidden:
```bash
python src/label/make_blind_sample.py                 # once; will not redraw
streamlit run src/label/label_app.py -- --blind        # saves to data/labelled/blind_labels.csv
python src/analysis/blind_agreement.py                 # agreement + Cohen's kappa -> results/
```
Both app modes record `seconds_spent` per question.

The labelling rules are in `docs/labelling_guide.md`, which the app shows in its sidebar. Edit the file and refresh the page to update it.

Shows one question at a time (Arabic right-to-left). Pick the problem type and who is asking (worker / employer / unclear), and type any facts the message states (amount, date, duration, employer). **Save & next** writes to `data/labelled/labelled.csv` straight away. Questions already saved are skipped next time, so you can stop whenever you like. The order is a fixed random shuffle, so whatever you have labelled is a random sample. The sidebar filters by language and law era and shows progress. `labelled.csv` has `text` and `problem_type` columns, so the baseline can read it directly.

### 6. RQ1: classifiers
All three train on the silver labels and are scored on the same test set (`src/models/common.py`): the fixed 20% of questions in `data/labelled/rq1_dev_split.csv` (91), never used for training.
- **human mode** (the RQ1 result): you label those 91 questions blind, and your labels replace the silver ones:
  ```
  streamlit run src/label/label_app.py -- --test
  ```
  The app hides the LLM's suggestion and the lawyers' answers, shows Arabic and English mixed in a fixed order, and saves to `data/labelled/test_labels.csv`. Fill in the type, who is asking and the facts (RQ2 uses them as the gold facts).
- **dev mode** (until then): silver labels on the test questions, a pipeline check only.
```bash
python src/models/tfidf_baseline.py      # TF-IDF + logistic regression
python src/models/embed.py               # BGE-M3 embeddings, once (downloads ~2.3 GB)
python src/models/bge_classifier.py      # BGE-M3 + logistic regression
python src/models/llm_classifier.py      # Qwen3-8B zero-shot, local and free (Ollama)
```
The LLM classifier needs Ollama running locally (`brew install ollama`, then `ollama serve` or the Ollama app) and `ollama pull qwen3:8b` (~5 GB). On an Apple M-series Mac with 16 GB+ it takes about 3 s per question.

**Arabic-focused model (Jais 2 8B Chat, Colab).** `python src/models/export_for_colab.py` writes the test questions to `colab/`. Run `notebooks/rq1_jais2_colab.ipynb` on a free T4 GPU (accept the model's terms on Hugging Face and add an `HF_TOKEN` Colab secret first), download `rq1_llm_jais-2-8b_cache.csv` into `results/`, then score it:
```bash
python src/models/llm_classifier.py --model jais-2-8b --cache-only
```
Falcon-H1-Arabic 7B was not used: it is not publicly available on Hugging Face (checked 23 Oct 2026).
Results: `results/rq1_<model>_<mode>_metrics.csv` (macro-F1 overall and per language), plus predictions and confusion matrix.

### 7. RQ3: split the law into articles
```bash
python src/law/split_articles.py
```
Reads `data/law/decree_law_33_2021_en.pdf` (MoHRE) and `data/law/decree_law_33_2021_ar.pdf` (u.ae); both are the Decree-Law consolidated with its amendments, plus Cabinet Resolution No. 1 of 2022, which is skipped. Output: `data/law/articles.csv` (`article, lang, title, amended, text, pages`), all 74 articles in each language. English text is read directly. The Arabic PDF's embedded text is broken, so it is read by OCR (Tesseract, Arabic model in `.venv/share/tessdata/`; needs `brew install tesseract`), with the OCR text cached in `data/law/ocr_ar/`. OCR makes small errors (for example ن/ت confusions in a few titles). The Arabic PDF has the original fines in Article 60, the English one the amended fines.

**Keyword table (draft).** `data/law/article_keywords.csv` has English and Arabic keywords per article (`article, title_en, keywords_en, keywords_ar, status`; `;` separates terms), written as workers phrase things. All rows are `status = draft` until you check them: hop 2 scores depend on them. Articles 1, 2, 70, 71 and 74 have no keywords (formal articles).

### 8. RQ3 hop 1: similar Q&As
```bash
python src/rag/hop1.py                     # problem type from the labels
python src/rag/hop1.py --pred results/rq1_bge_m3_haqqi_predictions.csv   # from a classifier
```
For each question, the 10 most similar Q&As (BGE-M3 cosine) with the same problem type, from the 451 scraped questions with a lawyer answer. Never the question itself or a copy of it (cosine >= 0.98, or question bodies >= 90% alike). Output: `data/rag/hop1_neighbors.csv`.

### 9. RQ3 hop 2: score law articles
```bash
python src/rag/hop2.py            # IDF-weighted keyword scores (--no-idf for plain counts)
```
Scores each article by its keywords found in the question alone (`one_hop`) and in the question plus its 10 hop-1 Q&As, questions and lawyers' answers (`two_hop`); keeps the top 3. Output: `data/rag/hop2_articles.csv` (with the matched keywords per article). Fill in `data/law/type_article_map_template.csv` and save it as `data/law/type_article_map.csv` to get recall@3 and hit@3 per language in `results/rq3_recall.csv`.

**For a fair comparison** run hop 1 with the classifier's predicted problem type (`hop1.py --pred ...`): with the gold type, two hops get the correct type for free through the same-type Q&As, and one hop does not.

### 10. Synthetic check (17 made-up cases)
```
python src/label/convert_synthetic_cases.py
```
Takes the Haqqi app's 50 made-up evaluation cases (`data/synthetic/cases.jsonl`) and keeps the 17 in Arabic or English that the law covers, with a hand-mapped problem type (check `MAPPING` in the script) and each case's own key articles (Decree-Law 33/2021 only). Output: `data/synthetic/synthetic_cases.csv`. The notebook reports problem-type accuracy and key-article recall@3 on them, **separately from the test results**: they are synthetic, so never the test set.

### 11. TF-IDF on your own file (optional)
Fill in `data/labelled/haqqi_50_template.csv` (columns: `id, text, lang, problem_type`), save it as `data/labelled/haqqi_50.csv`, then:
```bash
python src/models/tfidf_baseline.py --data data/labelled/haqqi_50.csv
```
Uses cross-validation when there is no `split` column. Reports macro-F1 overall and per language.

## Problem types
`unpaid_wages`, `passport_withheld`, `gratuity`, `unfair_termination`, `recruitment_fees`, `court_enforcement`, `other`

`court_enforcement` was added at the labelling checkpoint, after LLM pre-labelling showed that about 17% of questions (mostly Arabic) ask how to enforce a judgment or follow court procedure in a case already filed.

## Data and ethics notes
- Training labels (`data/labelled/prelabels.csv`, 452 scraped questions) were produced by Claude Opus 5.5 following `docs/labelling_guide.md` and are used as **silver labels**: they were not independently verified. The blind check (`blind_labels.csv`) was stopped after 3 questions, too few to report. The test set is the 91 held-out questions labelled blind by the author (`test_labels.csv`); their agreement with the silver labels measures silver-label quality.
- Data: public questions from Legal Advice Middle East (employment). Personal details are masked; names must be checked by hand.
- Polite scraping: 12 s delay, own user agent, no disallowed URLs.
- Generative AI was used to help write parts of this code (acknowledge as the subject outline requires).
