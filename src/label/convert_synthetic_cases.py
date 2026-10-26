"""
Turn the 50 made-up evaluation cases of the Haqqi app (data/synthetic/cases.jsonl)
into a small, separate "synthetic check" set for this project.

Usage (from the repo root):
    python src/label/convert_synthetic_cases.py

Kept: cases in Arabic or English that the app analyses (route "ready"): 17 cases.
Dropped: the other languages (Hindi, Urdu, ...), out-of-scope and need-info cases.

These cases are synthetic (no real people), so they are NEVER the test set:
they are an extra check, reported separately.

Output: data/synthetic/synthetic_cases.csv, utf-8-sig, with
  id, lang, text, issues (the app's own issue names), problem_type (our 7 types,
  mapped by MAPPING below + the labelling guide's "first mentioned" rule; check it),
  key_articles (the articles the case's key clauses come from) and
  supporting_articles (all articles the case lists), Decree-Law 33/2021 only,
  separated by ";". Cabinet Resolution 1/2022 articles are dropped: they are
  not in our article table.
"""
import json
import os

import pandas as pd

CASES = "data/synthetic/cases.jsonl"
OUT = "data/synthetic/synthetic_cases.csv"

# primary problem type, chosen by the guide: the main problem, or the first one mentioned.
# Written by hand from each story; check these before reporting.
MAPPING = {
    "TC-03": "gratuity",
    "TC-10": "unpaid_wages",
    "TC-11": "unpaid_wages",
    "TC-12": "other",                # visa overstay fine: not a labour-law issue
    "TC-18": "unpaid_wages",
    "TC-19": "unfair_termination",   # dismissed without notice
    "TC-20": "unfair_termination",   # termination mentioned first; also gratuity, wages
    "TC-21": "unpaid_wages",         # July salary mentioned first; also gratuity
    "TC-22": "other",                # annual leave
    "TC-23": "unpaid_wages",
    "N-07": "unfair_termination",    # termination first; also wages, gratuity
    "N-08": "gratuity",
    "N-15": "unpaid_wages",
    "N-16": "gratuity",
    "N-23": "gratuity",
    "N-24": "unpaid_wages",          # June/July salary first; also deduction, gratuity
    "N-27": "passport_withheld",     # passport first; also wages
}


def law_articles(refs) -> str:
    """'fdl33-2021:art22:cl2' -> '22'; Decree-Law 33/2021 only, no repeats, in order."""
    nums = []
    for ref in refs:
        law, art = ref.split(":")[:2]
        if law == "fdl33-2021" and int(art[3:]) not in nums:
            nums.append(int(art[3:]))
    return ";".join(map(str, nums))


def main():
    rows = []
    with open(CASES, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            if c["language"] not in ("ar", "en") or c["expected"]["route"] != "ready":
                continue
            rows.append({"id": c["id"], "lang": c["language"], "text": c["story"],
                         "issues": ";".join(c["expected"]["issues"]),
                         "problem_type": MAPPING[c["id"]],
                         "key_articles": law_articles(c["expected"]["key_clauses"]),
                         "supporting_articles": law_articles(c["expected"]["supporting_articles"]),
                         "source": "synthetic"})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{len(df)} cases -> {OUT}")
    print(pd.crosstab(df["problem_type"], df["lang"]).to_string())


if __name__ == "__main__":
    main()
