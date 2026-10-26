"""
Draw the random sample for the blind check of the LLM's training labels.

Usage (from the repo root):
    python src/label/make_blind_sample.py            # 30 Arabic + 30 English
    python src/label/make_blind_sample.py --n 40     # 40 of each

Then label them without seeing the LLM's suggestions:
    streamlit run src/label/label_app.py -- --blind
and compare:
    python src/analysis/blind_agreement.py

Input:  data/labelled/prelabels.csv  (the questions the LLM labelled)
        data/clean/questions_clean.csv (for the language)
Output: data/labelled/blind_sample.csv  (position, question_id, lang)

The sample is fixed once drawn: running this again does nothing unless you pass
--force, so the sample cannot be redrawn until the agreement looks good.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.label.build_queue import load_scraped  # noqa: E402
from src.label.prelabel import OUT as PRELABELS  # noqa: E402

SAMPLE = "data/labelled/blind_sample.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="questions per language")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true", help="overwrite an existing sample")
    args = ap.parse_args()

    if os.path.exists(SAMPLE) and not args.force:
        raise SystemExit(f"{SAMPLE} already exists; keeping it (use --force to redraw).")

    pre = pd.read_csv(PRELABELS, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    qs = load_scraped().merge(pre[pre["error"] == ""][["question_id"]], on="question_id")
    parts = [qs[qs["lang"] == lang].sample(n=args.n, random_state=args.seed) for lang in ("ar", "en")]
    # mix the two languages so the order gives nothing away
    sample = pd.concat(parts).sample(frac=1, random_state=args.seed)[["question_id", "lang"]]
    sample.insert(0, "position", range(1, len(sample) + 1))

    os.makedirs(os.path.dirname(SAMPLE), exist_ok=True)
    sample.to_csv(SAMPLE, index=False, encoding="utf-8-sig")
    print(f"{len(sample)} questions -> {SAMPLE}")
    print(sample["lang"].value_counts().to_string())


if __name__ == "__main__":
    main()
