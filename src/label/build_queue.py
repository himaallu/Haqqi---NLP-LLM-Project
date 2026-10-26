"""
Build the labelling queue: the order in which label_app.py shows questions.

Usage (from the repo root):
    python src/label/build_queue.py

Order:
  1. all Arabic questions: law_era 'current' first, then 'pre2022'
  2. 200 random English 'current' questions (--n-en to change)
  3. your Haqqi cases from data/labelled/haqqi_50.csv, in file order
Within groups 1 and 2 the order is a fixed random shuffle (--seed), so the
labels you have done at any point are a random sample of that group.

Input:  data/clean/questions_clean.csv, data/labelled/haqqi_50.csv (optional)
Output: data/labelled/queue.csv  (position, question_id, source, group)

haqqi_50.csv needs `id` and `text`; `lang` is optional (detected if empty).
If it does not exist yet, the queue is built without it: re-run this script
once you have saved it. With the same inputs and seed the scraped part of the
queue comes out the same, and labels already saved are never lost (the app
matches them by question_id).
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.preprocess.preprocess import preprocess  # noqa: E402

CLEAN = "data/clean/questions_clean.csv"
HAQQI = "data/labelled/haqqi_50.csv"
QUEUE = "data/labelled/queue.csv"


def load_scraped(path: str = CLEAN) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"question_id": str})
    df["source"] = "legaladviceme"
    return df


def load_haqqi(path: str = HAQQI) -> pd.DataFrame:
    """The Haqqi cases in the same columns as the scraped questions.
    question_id is 'haqqi_<id>' so it can never clash with a scraped id."""
    if not os.path.exists(path):
        return pd.DataFrame()
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    rows = []
    for r in raw.to_dict("records"):
        p = preprocess(r["text"])  # mask PII and detect language, as for scraped data
        rows.append({
            "question_id": f"haqqi_{r['id']}",
            "url": "", "title": "", "question": r["text"], "date": "", "answers": "[]",
            "full_text": r["text"], "text_masked": p["text_masked"],
            "lang": r.get("lang") or p["lang"],
            "law_era": "current",  # Haqqi cases are recent
            "source": "haqqi",
        })
    return pd.DataFrame(rows)


def all_questions() -> pd.DataFrame:
    return pd.concat([load_scraped(), load_haqqi()], ignore_index=True)


def build_queue(scraped: pd.DataFrame, haqqi: pd.DataFrame, n_en: int = 200, seed: int = 42) -> pd.DataFrame:
    def shuffled(df):
        return df.sample(frac=1, random_state=seed)

    ar = scraped[scraped["lang"] == "ar"]
    en_current = scraped[(scraped["lang"] == "en") & (scraped["law_era"] == "current")]
    groups = [
        ("ar_current", shuffled(ar[ar["law_era"] == "current"])),
        ("ar_pre2022", shuffled(ar[ar["law_era"] == "pre2022"])),
        ("en_current_sample", en_current.sample(n=min(n_en, len(en_current)), random_state=seed)),
        ("haqqi", haqqi),
    ]
    parts = [pd.DataFrame({"question_id": g["question_id"], "source": g["source"], "group": name})
             for name, g in groups if len(g)]
    queue = pd.concat(parts, ignore_index=True)
    queue.insert(0, "position", range(1, len(queue) + 1))
    return queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-en", type=int, default=200, help="number of English 'current' questions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    scraped, haqqi = load_scraped(), load_haqqi()
    queue = build_queue(scraped, haqqi, args.n_en, args.seed)
    os.makedirs(os.path.dirname(QUEUE), exist_ok=True)
    queue.to_csv(QUEUE, index=False, encoding="utf-8-sig")

    print(f"{len(queue)} questions -> {QUEUE}")
    print(queue["group"].value_counts(sort=False).to_string())
    if haqqi.empty:
        print(f"\nNo {HAQQI} yet: Haqqi cases not added. Save it, then run this again.")
    left_out = scraped[~scraped["lang"].isin(["ar", "en"])]
    if len(left_out):
        print(f"\nNot queued (language neither ar nor en): {json.dumps(left_out['question_id'].tolist())}")


if __name__ == "__main__":
    main()
