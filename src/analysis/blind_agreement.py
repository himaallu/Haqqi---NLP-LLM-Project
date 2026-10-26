"""
How well do the LLM's training labels agree with blind human labels?

Usage (from the repo root), after labelling the sample:
    python src/analysis/blind_agreement.py           # blind labels (label_app.py --blind)
    python src/analysis/blind_agreement.py --check   # labels made with suggestions shown (--check)
Only the blind version measures independent agreement; the --check version is
reported as "verified with the LLM's suggestions visible".

Input:  data/labelled/blind_labels.csv or check_labels.csv  (your labels)
        data/labelled/prelabels.csv                           (the LLM's labels)
Output: results/blind_agreement.csv or check_agreement.csv    (one row per measure and group)
        results/blind_confusion.csv or check_confusion.csv    (problem_type: yours in rows, LLM's in columns)

Measures
  - problem_type and asker: percent agreement and Cohen's kappa (agreement
    corrected for chance), overall and per language.
  - facts: percent of questions where both give exactly the same text (both
    empty counts as agreeing), and how often each side filled the box.
Also prints the median seconds you spent per question.
"""
import argparse
import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score

BLIND = "data/labelled/blind_labels.csv"
CHECKED = "data/labelled/check_labels.csv"
SAMPLE = "data/labelled/blind_sample.csv"
PRELABELS = "data/labelled/prelabels.csv"
OUT = "results/{mode}_agreement.csv"
CONFUSION = "results/{mode}_confusion.csv"
FACTS = ["amount", "date_mentioned", "duration", "employer"]


def agreement(df: pd.DataFrame, field: str) -> dict:
    human, llm = df[field], df[f"llm_{field}"]
    # kappa is undefined when both sides use a single label only
    kappa = cohen_kappa_score(human, llm) if len(set(human) | set(llm)) > 1 else float("nan")
    return {"n": len(df), "agreement_pct": round(100 * (human == llm).mean(), 1), "kappa": round(kappa, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="use check_labels.csv (suggestions were visible)")
    args = ap.parse_args()
    mode, path = ("check", CHECKED) if args.check else ("blind", BLIND)
    out_path, confusion_path = OUT.format(mode=mode), CONFUSION.format(mode=mode)
    print("Mode:", "CHECK WITH SUGGESTIONS VISIBLE (not blind)" if args.check else "BLIND")

    if not os.path.exists(path):
        raise SystemExit(f"No labels yet ({path}). Label the sample first:\n"
                         f"    streamlit run src/label/label_app.py -- --{mode}")
    blind = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype=str)
    if len(blind) < len(sample):
        print(f"Note: only {len(blind)} of {len(sample)} sample questions labelled so far; "
              "the numbers below are partial.\n")
    # drop the llm_ columns the app saved (empty in blind mode) before joining the real ones
    blind = blind.drop(columns=[c for c in blind if c.startswith("llm_")])
    pre = pd.read_csv(PRELABELS, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = blind.merge(pre, on="question_id", how="inner")
    print(f"{len(df)} of your labels matched to LLM labels")

    rows = []
    for field in ("problem_type", "asker"):
        rows.append({"measure": field, "group": "all", **agreement(df, field)})
        for lang, g in df.groupby("lang"):
            rows.append({"measure": field, "group": lang, **agreement(g, field)})
    for f in FACTS:
        same = df[f].str.strip() == df[f"llm_{f}"].str.strip()
        rows.append({"measure": f"fact:{f}", "group": "all", "n": len(df),
                     "agreement_pct": round(100 * same.mean(), 1), "kappa": float("nan"),
                     "human_filled": int((df[f] != "").sum()), "llm_filled": int((df[f"llm_{f}"] != "").sum())})

    out = pd.DataFrame(rows)
    out.insert(0, "mode", "suggestions_visible" if args.check else "blind")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    confusion = pd.crosstab(df["problem_type"], df["llm_problem_type"], rownames=["human"], colnames=["llm"])
    confusion.to_csv(confusion_path, encoding="utf-8-sig")

    print(out.to_string(index=False))
    print("\nproblem_type, yours (rows) vs LLM (columns):")
    print(confusion.to_string())
    if "seconds_spent" in df and (df["seconds_spent"] != "").any():
        secs = pd.to_numeric(df["seconds_spent"], errors="coerce")
        print(f"\nseconds per question: median {secs.median():.0f}, min {secs.min():.0f}")
    print(f"\n-> {out_path}, {confusion_path}")


if __name__ == "__main__":
    main()
