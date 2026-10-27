"""
Label the 91 RQ1 test questions in Excel instead of the Streamlit app.

Run from the repo root:
    python src/label/test_sheet.py export   # writes data/labelled/test_to_label.xlsx
    python src/label/test_sheet.py import   # reads it back into data/labelled/test_labels.csv

Export: the test questions in the same fixed mixed order as `label_app.py --test`,
masked text only (no lawyers' answers, no LLM suggestions, no URL), with dropdowns
for problem_type and asker and empty boxes for the facts. A second sheet holds
the labelling guide.

Import: every row with a problem_type and an asker is saved to test_labels.csv in
the same columns the app writes, so the rest of the pipeline does not change.
Rows without both are skipped and listed.
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.label.build_queue import all_questions  # noqa: E402

TEST_SPLIT = "data/labelled/rq1_dev_split.csv"
SHEET = "data/labelled/test_to_label.xlsx"
LABELS = "data/labelled/test_labels.csv"
GUIDE = "docs/labelling_guide.md"
SEED = 42  # same as label_app.py, so the order matches the app

PROBLEM_TYPES = ["unpaid_wages", "passport_withheld", "gratuity",
                 "unfair_termination", "recruitment_fees", "court_enforcement", "other"]
ASKERS = ["worker", "employer", "unclear"]
# columns you fill in, in sheet order
FILL = ["problem_type", "asker", "amount", "date_mentioned", "duration", "employer", "notes"]
# same columns as label_app.py writes
LABEL_COLS = ["question_id", "text", "lang", "law_era", "problem_type", "asker",
              "amount", "date_mentioned", "duration", "employer", "notes", "source",
              "llm_problem_type", "llm_asker", "labelled_at", "seconds_spent"]


def test_questions() -> pd.DataFrame:
    """The test questions in the app's fixed mixed-language order."""
    ids = pd.read_csv(TEST_SPLIT, dtype=str)[["question_id"]].sample(frac=1, random_state=SEED)
    return ids.merge(all_questions(), on="question_id", how="inner").reset_index(drop=True)


def export():
    qs = test_questions()
    wb = Workbook()
    ws = wb.active
    ws.title = "label"
    header = ["n", "question_id", "lang", "question"] + FILL
    ws.append(header)
    for i, q in qs.iterrows():
        ws.append([i + 1, q["question_id"], q["lang"], q["text_masked"]] + [""] * len(FILL))

    # layout: wide wrapped question column, Arabic right-to-left, yellow boxes to fill
    widths = {"A": 5, "B": 11, "C": 6, "D": 90, "E": 20, "F": 11, "G": 14, "H": 14, "I": 12, "J": 18, "K": 30}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for cell in ws[1]:
        cell.font = Font(bold=True)
    fill = PatternFill("solid", fgColor="FFF2CC")
    for row in ws.iter_rows(min_row=2):
        rtl = row[2].value == "ar"
        row[3].alignment = Alignment(wrap_text=True, vertical="top",
                                     horizontal="right" if rtl else "left", readingOrder=2 if rtl else 1)
        for cell in row[:3]:
            cell.alignment = Alignment(vertical="top")
        for cell in row[4:]:
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "E2"

    # dropdowns, so only valid labels can be typed
    last = len(qs) + 1
    for col, options in (("E", PROBLEM_TYPES), ("F", ASKERS)):
        dv = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=True)
        dv.add(f"{col}2:{col}{last}")
        ws.add_data_validation(dv)

    guide = wb.create_sheet("guide")
    guide.column_dimensions["A"].width = 120
    with open(GUIDE, encoding="utf-8") as f:
        for line in f.read().splitlines():
            guide.append([line])

    wb.save(SHEET)
    print(f"Wrote {SHEET}: {len(qs)} questions ({qs['lang'].value_counts().to_dict()})")


def import_sheet():
    wb = load_workbook(SHEET, read_only=True)
    rows = list(wb["label"].values)
    sheet = pd.DataFrame(rows[1:], columns=rows[0]).fillna("").astype(str)
    sheet = sheet.apply(lambda c: c.str.strip())
    sheet["question_id"] = sheet["question_id"].str.removesuffix(".0")

    bad = sheet[~sheet["problem_type"].isin(PROBLEM_TYPES) | ~sheet["asker"].isin(ASKERS)]
    done = sheet.drop(bad.index)

    qs = test_questions().set_index("question_id")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = pd.DataFrame({
        "question_id": done["question_id"],
        "text": done["question_id"].map(qs["text_masked"]),
        "lang": done["question_id"].map(qs["lang"]),
        "law_era": done["question_id"].map(qs["law_era"]),
        **{c: done[c] for c in FILL},
        "source": done["question_id"].map(qs["source"]),
        "llm_problem_type": "", "llm_asker": "",  # labelled blind: no suggestions
        "labelled_at": now, "seconds_spent": "",
    })
    out.reindex(columns=LABEL_COLS).to_csv(LABELS, index=False, encoding="utf-8-sig")
    print(f"Saved {len(out)} labels to {LABELS}")
    if len(bad):
        print(f"Skipped {len(bad)} rows without a valid problem_type and asker: n =",
              ", ".join(bad["n"]))


if __name__ == "__main__":
    {"export": export, "import": import_sheet}[sys.argv[1] if len(sys.argv) > 1 else "export"]()
