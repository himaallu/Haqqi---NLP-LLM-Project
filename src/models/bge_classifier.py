"""
RQ1 model 2: BGE-M3 embeddings + logistic regression.

Usage (from the repo root):
    python src/models/embed.py            # once (and after adding Haqqi cases)
    python src/models/bge_classifier.py

Uses the same train/test split and scoring as the other classifiers (common.py).
The embedding model is frozen; only the logistic regression on top is trained.
"""
import os
import sys

from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.common import get_split, report  # noqa: E402
from src.models.embed import get_embeddings, update_cache  # noqa: E402


def main():
    train, test, mode = get_split()
    print(f"RQ1 split ({mode}): {len(train)} train, {len(test)} test")
    update_cache()  # embeds any new questions (e.g. Haqqi cases)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0)
    clf.fit(get_embeddings(train["question_id"].tolist()), train["problem_type"])
    pred = clf.predict(get_embeddings(test["question_id"].tolist()))
    report("bge_m3", mode, test, list(pred))


if __name__ == "__main__":
    main()
