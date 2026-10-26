"""
Export the RQ1 test questions for the Colab notebook (Arabic-focused model, Jais 2 8B).

Usage (from the repo root):
    python src/models/export_for_colab.py

Writes colab/rq1_test.csv (question_id, lang, text) for the current test split
(Haqqi cases if available, otherwise the dev split) and copies the labelling
guide to colab/labelling_guide.md. Upload both to Colab with
notebooks/rq1_jais2_colab.ipynb. Download the notebook's output,
rq1_llm_jais-2-8b_cache.csv, into results/ and score it with:
    python src/models/llm_classifier.py --model jais-2-8b --cache-only
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.common import get_split  # noqa: E402

OUT_DIR = "colab"


def main():
    _, test, mode = get_split()
    os.makedirs(OUT_DIR, exist_ok=True)
    test[["question_id", "lang", "text"]].to_csv(f"{OUT_DIR}/rq1_test.csv", index=False, encoding="utf-8-sig")
    shutil.copy("docs/labelling_guide.md", f"{OUT_DIR}/labelling_guide.md")
    print(f"{len(test)} test questions ({mode} mode) -> {OUT_DIR}/rq1_test.csv, {OUT_DIR}/labelling_guide.md")


if __name__ == "__main__":
    main()
