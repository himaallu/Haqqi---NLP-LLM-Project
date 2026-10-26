"""
Traditional baseline: TF-IDF + logistic regression for problem-type classification.

Works on a small set (your 50 Haqqi cases) and on the full labelled data.

Input CSV needs these columns:
    text          the worker's message (raw; preprocessing is applied here)
    problem_type  the label
    split         optional: 'train' / 'test'. If missing, 5-fold cross-validation is used.

Usage:
    python src/models/tfidf_baseline.py            # RQ1: shared split from common.py
    python src/models/tfidf_baseline.py --data data/labelled/haqqi_50.csv
    python src/models/tfidf_baseline.py --data data/labelled/labelled.csv   # with a split column

Features: word 1-2 grams + character 2-5 grams. Character n-grams help with Arabic
spelling variants, prefixes like "ال" / "و", and typos in informal messages.

Reports macro-F1 overall and per language, and saves a confusion matrix CSV
to results/tfidf_confusion.csv.
"""
import argparse
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.preprocess.preprocess import preprocess  # noqa: E402
from src.models.common import get_split, report as rq1_report  # noqa: E402


def build_model() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0)
    return Pipeline([("tfidf", features), ("clf", clf)])


def report(y_true, y_pred, langs, labels):
    print(f"\nMacro-F1 overall: {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
    for lang in sorted(set(langs)):
        mask = [l == lang for l in langs]
        yt = [y for y, m in zip(y_true, mask) if m]
        yp = [y for y, m in zip(y_pred, mask) if m]
        print(f"Macro-F1 [{lang}] (n={len(yt)}): {f1_score(yt, yp, average='macro', zero_division=0):.3f}")
    print("\n" + classification_report(y_true, y_pred, labels=labels, zero_division=0))
    os.makedirs("results", exist_ok=True)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    cm.to_csv("results/tfidf_confusion.csv")
    print("Confusion matrix saved to results/tfidf_confusion.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="your own CSV (text, problem_type, optional split); "
                                   "leave out to use the shared RQ1 split")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    if not args.data:  # RQ1: same train/test as the other classifiers
        train, test, mode = get_split()
        print(f"RQ1 split ({mode}): {len(train)} train, {len(test)} test")
        model = build_model()
        model.fit(train["text"].map(lambda t: preprocess(t)["text_norm"]), train["problem_type"])
        pred = model.predict(test["text"].map(lambda t: preprocess(t)["text_norm"]))
        rq1_report("tfidf", mode, test, list(pred))
        return

    df = pd.read_csv(args.data, encoding="utf-8-sig").dropna(subset=["text", "problem_type"])
    pre = df["text"].apply(preprocess).apply(pd.Series)
    df["text_norm"], df["lang"] = pre["text_norm"], pre["lang"]
    labels = sorted(df["problem_type"].unique())
    print(f"{len(df)} examples, {len(labels)} classes: {df['problem_type'].value_counts().to_dict()}")

    model = build_model()
    if "split" in df.columns and set(df["split"]) >= {"train", "test"}:
        train, test = df[df["split"] == "train"], df[df["split"] == "test"]
        model.fit(train["text_norm"], train["problem_type"])
        pred = model.predict(test["text_norm"])
        report(list(test["problem_type"]), list(pred), list(test["lang"]), labels)
    else:
        folds = min(args.folds, df["problem_type"].value_counts().min())
        if folds < 2:
            raise SystemExit("Every class needs at least 2 examples for cross-validation.")
        print(f"No train/test split column: using {folds}-fold cross-validation")
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        pred = cross_val_predict(model, df["text_norm"], df["problem_type"], cv=cv)
        report(list(df["problem_type"]), list(pred), list(df["lang"]), labels)


if __name__ == "__main__":
    main()
