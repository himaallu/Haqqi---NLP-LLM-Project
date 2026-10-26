"""
Shared data split and scoring for the RQ1 classifiers (TF-IDF, BGE-M3, LLM).

Every classifier trains and is scored on exactly the same questions:
  train: the silver labels on scraped questions (data/labelled/labelled.csv,
         LLM-made following docs/labelling_guide.md, not verified)
  test:  "human" mode - the same fixed 20% of questions (data/labelled/rq1_dev_split.csv),
         with your own blind labels from the app's --test mode
         (data/labelled/test_labels.csv) instead of the silver ones (the RQ1 result);
         "haqqi" mode - your Haqqi cases in data/labelled/haqqi_50.csv, once the
         file exists and has problem_type filled in (the real RQ1 result);
         "dev"  mode - until then, a fixed stratified 20% of the silver labels
         (data/labelled/rq1_dev_split.csv). Dev scores only show the code works
         and how well a model copies the LLM's labels; do not report them as RQ1.

Results go to results/rq1_<model>_<mode>_{metrics,predictions,confusion}.csv.
"""
import os
import sys

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.preprocess.preprocess import preprocess  # noqa: E402

SILVER = "data/labelled/labelled.csv"
HAQQI = "data/labelled/haqqi_50.csv"
DEV_SPLIT = "data/labelled/rq1_dev_split.csv"
TEST_LABELS = "data/labelled/test_labels.csv"  # your blind labels for the DEV_SPLIT questions
RESULTS = "results"

PROBLEM_TYPES = ["unpaid_wages", "passport_withheld", "gratuity", "unfair_termination",
                 "recruitment_fees", "court_enforcement", "other"]
# its text is the site's search summary and contains the lawyer's answer (label leak)
EXCLUDE = {"44267"}


def load_silver() -> pd.DataFrame:
    df = pd.read_csv(SILVER, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = df[(df["source"] == "legaladviceme") & ~df["question_id"].isin(EXCLUDE)]
    return df[["question_id", "text", "lang", "law_era", "problem_type"]].reset_index(drop=True)


def load_haqqi() -> pd.DataFrame:
    """Your labelled Haqqi cases, or an empty frame if not available yet."""
    if not os.path.exists(HAQQI):
        return pd.DataFrame()
    raw = pd.read_csv(HAQQI, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    raw = raw[raw["problem_type"].isin(PROBLEM_TYPES) & (raw["text"].str.strip() != "")]
    rows = []
    for r in raw.to_dict("records"):
        p = preprocess(r["text"])  # same masking as the scraped questions
        rows.append({"question_id": f"haqqi_{r['id']}", "text": p["text_masked"],
                     "lang": r.get("lang") or p["lang"], "law_era": "current",
                     "problem_type": r["problem_type"]})
    return pd.DataFrame(rows)


def get_split() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(train, test, mode): 'haqqi' if the Haqqi test set exists, else 'human' if you
    labelled the test questions, else 'dev'."""
    silver, haqqi = load_silver(), load_haqqi()
    if len(haqqi):
        return silver, haqqi, "haqqi"
    if not os.path.exists(DEV_SPLIT):  # draw once, then always reuse
        _, test_ids = train_test_split(silver["question_id"], test_size=0.2,
                                       stratify=silver["problem_type"], random_state=42)
        os.makedirs(os.path.dirname(DEV_SPLIT), exist_ok=True)
        pd.DataFrame({"question_id": sorted(test_ids)}).to_csv(DEV_SPLIT, index=False)
    test_ids = set(pd.read_csv(DEV_SPLIT, dtype=str)["question_id"])
    is_test = silver["question_id"].isin(test_ids)
    train, test = silver[~is_test].reset_index(drop=True), silver[is_test].reset_index(drop=True)
    if os.path.exists(TEST_LABELS):  # your labels replace the silver ones on the test questions
        human = pd.read_csv(TEST_LABELS, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        human = human[human["problem_type"].isin(PROBLEM_TYPES)].set_index("question_id")["problem_type"]
        if len(human) < len(test):
            print(f"note: {len(test) - len(human)} test questions have no human label yet and are left out")
        test = test[test["question_id"].isin(human.index)].copy()
        test["problem_type"] = test["question_id"].map(human)
        return train, test.reset_index(drop=True), "human"
    return train, test, "dev"


def report(model: str, mode: str, test: pd.DataFrame, pred: list) -> pd.DataFrame:
    """Print and save macro-F1 overall and per language, predictions and confusion matrix."""
    y_true, langs = test["problem_type"].tolist(), test["lang"].tolist()
    print(f"\n=== {model} | mode: {mode}" + (" (silver held-out: NOT the RQ1 result)" if mode == "dev" else ""))
    rows = []
    for group in ["all"] + sorted(set(langs)):
        idx = [i for i, l in enumerate(langs) if group == "all" or l == group]
        yt, yp = [y_true[i] for i in idx], [pred[i] for i in idx]
        rows.append({"model": model, "mode": mode, "group": group, "n": len(idx),
                     "macro_f1": round(f1_score(yt, yp, average="macro", zero_division=0), 3),
                     "accuracy": round(sum(a == b for a, b in zip(yt, yp)) / len(idx), 3)})
    metrics = pd.DataFrame(rows)
    print(metrics.to_string(index=False))
    print("\n" + classification_report(y_true, pred, labels=PROBLEM_TYPES, zero_division=0))

    os.makedirs(RESULTS, exist_ok=True)
    base = f"{RESULTS}/rq1_{model}_{mode}"
    metrics.to_csv(f"{base}_metrics.csv", index=False, encoding="utf-8-sig")
    test.assign(pred=pred)[["question_id", "lang", "problem_type", "pred"]].to_csv(
        f"{base}_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(confusion_matrix(y_true, pred, labels=PROBLEM_TYPES),
                 index=PROBLEM_TYPES, columns=PROBLEM_TYPES).to_csv(f"{base}_confusion.csv")
    print(f"-> {base}_metrics.csv / _predictions.csv / _confusion.csv")
    return metrics
