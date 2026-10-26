"""
How often do lawyers' answers link to Federal Decree-Law No. 33 of 2021?

Usage (from the repo root):
    python src/analysis/answer_laws_stats.py

Input:  data/clean/questions_clean.csv   (answer_laws column from parse_pages.py)
Output: results/answer_laws_stats.csv

For each language x law_era (plus 'all' rows per language and overall) it counts
questions with at least one answer link to:
  - Decree-Law 33/2021 (law id 159 in the site's /legislation/<id>/ links)
  - any law at all
  - only the old Federal Law 8/1980 (id 121), i.e. 8/1980 but not 33/2021
A link is not the same as a citation: lawyers also name laws without linking,
and the site links some phrases (e.g. "end-of-service") automatically.
"""
import json
import os

import pandas as pd

CLEAN = "data/clean/questions_clean.csv"
OUT = "results/answer_laws_stats.csv"
DL_33_2021, FL_8_1980 = "159", "121"


def count(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "questions": n,
        "with_33_2021": int(df["dl33"].sum()),
        "share_33_2021_pct": round(100 * df["dl33"].mean(), 1) if n else 0.0,
        "any_law_link": int(df["any_law"].sum()),
        "old_1980_only": int(df["old_only"].sum()),
    }


def main():
    df = pd.read_csv(CLEAN, encoding="utf-8-sig")
    law_ids = df["answer_laws"].map(lambda s: {x["law_id"] for x in json.loads(s)})
    df["dl33"] = law_ids.map(lambda ids: DL_33_2021 in ids)
    df["any_law"] = law_ids.map(bool)
    df["old_only"] = law_ids.map(lambda ids: FL_8_1980 in ids and DL_33_2021 not in ids)

    rows = []
    for lang, g in df.groupby("lang"):
        for era, h in g.groupby("law_era"):
            rows.append({"lang": lang, "law_era": era, **count(h)})
        rows.append({"lang": lang, "law_era": "all", **count(g)})
    rows.append({"lang": "all", "law_era": "all", **count(df)})

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
