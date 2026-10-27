"""
LLM pre-labelling of the training set, for you to check in label_app.py.

Claude reads each scraped question in the labelling queue and suggests
problem_type, asker, the four facts and Notes, following the same
docs/labelling_guide.md that you label with. The app pre-fills its boxes with
these suggestions; you check and correct every one before saving.

Only scraped questions (source = legaladviceme) are pre-labelled. Your Haqqi
cases are the test set and are never sent: you label them blind, so the LLM
arm of RQ1 is not scored against labels the LLM itself suggested.

Usage (from the repo root; needs ANTHROPIC_API_KEY in the environment):
    python src/label/prelabel.py --dry-run     # print one prompt, no API call
    python src/label/prelabel.py --limit 5     # try a few first
    python src/label/prelabel.py               # the rest (resumes)

Input:  data/labelled/queue.csv, data/clean/questions_clean.csv, docs/labelling_guide.md
Output: data/labelled/prelabels.csv  (one row per question, saved after each call)
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

try:
    import anthropic
except ImportError:  # label_app.py imports OUT from here and must run without it
    anthropic = None
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.label.build_queue import QUEUE, all_questions  # noqa: E402

GUIDE = "docs/labelling_guide.md"
OUT = "data/labelled/prelabels.csv"
MODEL = "claude-opus-5-5"

PROBLEM_TYPES = ["unpaid_wages", "passport_withheld", "gratuity",
                 "unfair_termination", "recruitment_fees", "court_enforcement", "other"]
ASKERS = ["worker", "employer", "unclear"]
FACTS = ["amount", "date_mentioned", "duration", "employer"]
FIELDS = ["problem_type", "asker", *FACTS, "notes"]
OUT_COLS = ["question_id", *[f"llm_{f}" for f in FIELDS], "model", "served_by", "error", "prelabelled_at"]

# JSON schema the answer must follow (structured outputs: always valid JSON)
SCHEMA = {
    "type": "object",
    "properties": {
        "problem_type": {"type": "string", "enum": PROBLEM_TYPES},
        "asker": {"type": "string", "enum": ASKERS},
        **{f: {"type": "string"} for f in FACTS},
        "notes": {"type": "string"},
    },
    "required": FIELDS,
    "additionalProperties": False,
}

INSTRUCTIONS = """You label questions that people in the UAE posted about employment problems.
Follow the labelling guide below exactly; it is the same guide the human labeller uses.

Return:
- problem_type and asker: one of the allowed values.
- amount, date_mentioned, duration, employer: the exact words from the question,
  in the question's own language. Use an empty string when the question does not
  state it. Never translate, calculate or guess.
- notes: only the tags and short notes the guide asks for (empty string if none).

<labelling_guide>
{guide}
</labelling_guide>"""


def system_prompt() -> str:
    with open(GUIDE, encoding="utf-8") as f:
        return INSTRUCTIONS.format(guide=f.read())


def suggest(client: "anthropic.Anthropic", system: str, text: str) -> tuple[dict, str]:
    """Ask Claude for labels for one question. Returns (labels, model that answered)."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        # the instructions + guide are the same for every question: cache them
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"<question>\n{text}\n</question>"}],
        output_config={
            "effort": "low",  # a classification task; low effort keeps it cheap
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        # if Claude's safety filter declines, the API retries on a fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"refused ({response.stop_details.category if response.stop_details else 'no category'})")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("ran out of tokens")
    text_block = next(b.text for b in response.content if b.type == "text")
    return json.loads(text_block), response.model


def load_done(path: str = OUT) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["question_id"] for r in csv.DictReader(f) if not r["error"]}


def append_row(row: dict, path: str = OUT) -> None:
    """Add one result; a question that failed before is replaced when it succeeds."""
    old = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False) if os.path.exists(path) \
        else pd.DataFrame(columns=OUT_COLS)
    old = old[old["question_id"] != row["question_id"]]
    new = pd.concat([old, pd.DataFrame([row])], ignore_index=True).reindex(columns=OUT_COLS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    new.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def todo_questions() -> pd.DataFrame:
    """Scraped questions in queue order that have no suggestion yet."""
    queue = pd.read_csv(QUEUE, encoding="utf-8-sig", dtype=str)[["question_id"]]
    qs = queue.merge(all_questions(), on="question_id", how="inner")  # keeps queue order
    qs = qs[qs["source"] == "legaladviceme"]  # never the Haqqi test cases
    return qs[~qs["question_id"].isin(load_done())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after this many questions (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="print the prompt for one question; no API call")
    args = ap.parse_args()

    todo = todo_questions()
    if args.limit:
        todo = todo.head(args.limit)
    system = system_prompt()
    print(f"{len(todo)} questions to pre-label with {MODEL}")
    if todo.empty:
        return

    if args.dry_run:
        print("\n--- system ---\n" + system + "\n--- user ---\n<question>\n" + todo.iloc[0]["text_masked"] + "\n</question>")
        return

    if anthropic is None:
        raise SystemExit("The anthropic package is missing: pip install anthropic")
    has_profile = os.path.isdir(os.path.expanduser("~/.config/anthropic"))  # from `ant auth login`
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or has_profile):
        raise SystemExit("No API credentials: run  export ANTHROPIC_API_KEY=...  in this terminal first.")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    for i, q in enumerate(todo.itertuples(), 1):
        row = {"question_id": q.question_id, "model": MODEL,
               "prelabelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "error": ""}
        try:
            labels, served_by = suggest(client, system, q.text_masked)
            row.update({f"llm_{f}": labels.get(f, "") for f in FIELDS}, served_by=served_by)
            print(f"[{i}/{len(todo)}] {q.question_id} {q.lang}: {labels['problem_type']} / {labels['asker']}")
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise SystemExit(f"Stopped: {e.message}. Is ANTHROPIC_API_KEY set and valid?")
        except anthropic.RateLimitError as e:
            raise SystemExit(f"Stopped: rate limited ({e.message}). Run again later; it resumes.")
        except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError, json.JSONDecodeError) as e:
            row["error"] = str(e)[:200]
            print(f"[{i}/{len(todo)}] {q.question_id}: error {row['error']}")
        append_row(row)


if __name__ == "__main__":
    main()
