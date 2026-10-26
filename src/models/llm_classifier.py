"""
RQ1 model 3: LLM prompting with a local open model through Ollama (free, offline).

Default model: Qwen3-8B (`ollama pull qwen3:8b`, about 5 GB; runs on an Apple
M-series Mac with 16 GB+ memory at roughly 1-5 s per question).

Usage (from the repo root; the Ollama app or `ollama serve` must be running):
    python src/models/llm_classifier.py --dry-run          # count the questions
    python src/models/llm_classifier.py                    # qwen3:8b
    python src/models/llm_classifier.py --model other:tag  # any other Ollama model
    python src/models/llm_classifier.py --model jais-2-8b --cache-only
        # score answers made elsewhere (the Colab notebook), no Ollama needed

Zero-shot: the prompt is the labelling guide (docs/labelling_guide.md) plus the
question; the answer is forced to one of the problem types by a JSON schema.
Qwen3's "thinking" mode is switched off and temperature is 0, so answers are
repeatable. Same split and scoring as the other classifiers (common.py).
Answers are cached in results/rq1_llm_<model>_cache.csv, so a re-run only asks
for questions not answered yet; each question is tried up to 3 times.

Note: in "dev" mode the test labels are the silver (Claude-made) labels, so the
score shows agreement with Claude, not correctness. Only "haqqi" mode is RQ1.
"""
import argparse
import json
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.common import PROBLEM_TYPES, RESULTS, get_split, report  # noqa: E402

GUIDE = "docs/labelling_guide.md"
TRIES = 3

SCHEMA = {
    "type": "object",
    "properties": {"problem_type": {"type": "string", "enum": PROBLEM_TYPES}},
    "required": ["problem_type"],
}

INSTRUCTIONS = """You classify questions that people in the UAE posted about employment problems.
Read the question and choose its problem_type, following the labelling guide below
exactly. Answer with JSON: {{"problem_type": "<one of the allowed values>"}}.

<labelling_guide>
{guide}
</labelling_guide>"""


def system_prompt() -> str:
    with open(GUIDE, encoding="utf-8") as f:
        return INSTRUCTIONS.format(guide=f.read())


def cache_path(model: str) -> str:
    return f"{RESULTS}/rq1_llm_{re.sub(r'[^A-Za-z0-9.]+', '-', model)}_cache.csv"


def load_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return dict(zip(df["question_id"], df["pred"]))


def save_cache(path: str, cache: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    tmp = path + ".tmp"
    pd.DataFrame({"question_id": list(cache), "pred": list(cache.values())}).to_csv(
        tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def classify(model: str, system: str, text: str) -> str:
    """One question -> problem_type, retrying up to TRIES times."""
    import ollama

    for attempt in range(1, TRIES + 1):
        try:
            r = ollama.chat(model=model, think=False, format=SCHEMA, options={"temperature": 0},
                            messages=[{"role": "system", "content": system},
                                      {"role": "user", "content": f"<question>\n{text}\n</question>"}])
            label = json.loads(r.message.content)["problem_type"]
            if label in PROBLEM_TYPES:
                return label
            raise ValueError(f"unknown label {label!r}")
        except Exception as e:  # connection error, bad JSON, unknown label
            if attempt == TRIES:
                raise
            print(f"   retry {attempt} after: {str(e)[:80]}")
            time.sleep(2 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b", help="Ollama model tag")
    ap.add_argument("--dry-run", action="store_true", help="show how many questions need an answer")
    ap.add_argument("--cache-only", action="store_true",
                    help="only score answers already in the cache (e.g. from the Colab notebook)")
    args = ap.parse_args()

    _, test, mode = get_split()
    path = cache_path(args.model)
    cache = load_cache(path)
    todo = test[~test["question_id"].isin(cache)]
    print(f"RQ1 split ({mode}): {len(test)} test questions, {len(todo)} to classify with {args.model}")
    if args.dry_run:
        return

    if len(todo) and not args.cache_only:
        import ollama
        try:
            ollama.show(args.model)
        except Exception:
            raise SystemExit(f"Ollama is not running or {args.model} is not pulled: "
                             f"start the Ollama app (or `ollama serve`), then `ollama pull {args.model}`.")
        system = system_prompt()
        for i, q in enumerate(todo.itertuples(), 1):
            try:
                cache[q.question_id] = classify(args.model, system, q.text)
            except Exception as e:  # keep going; this question is retried on the next run
                print(f"[{i}/{len(todo)}] {q.question_id}: failed ({str(e)[:100]})")
                continue
            save_cache(path, cache)
            print(f"[{i}/{len(todo)}] {q.question_id} {q.lang}: {cache[q.question_id]}")

    missing = test[~test["question_id"].isin(cache)]
    if len(missing):
        hint = (f"put the notebook's output at {path}" if args.cache_only else "run again")
        raise SystemExit(f"{len(missing)} questions still have no prediction; {hint}.")
    name = "llm_" + re.sub(r"[^A-Za-z0-9.]+", "-", args.model)
    report(name, mode, test, [cache[q] for q in test["question_id"]])


if __name__ == "__main__":
    main()
