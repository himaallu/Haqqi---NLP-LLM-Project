"""
Labelling app: label the scraped questions one at a time (Streamlit).

Run from the repo root:
    streamlit run src/label/label_app.py

Input:  data/clean/questions_clean.csv   (from preprocess.py)
Output: data/labelled/labelled.csv       (saved after every question)

For each question you choose
  - problem_type: unpaid_wages, passport_withheld, gratuity, unfair_termination,
                  recruitment_fees, court_enforcement, other
  - asker:        worker, employer, unclear
and can fill in the facts the message states (amount, date, duration, employer).
Leave a fact box empty if the message does not state it.

Resume: questions already in labelled.csv are skipped when the app starts, so
you can stop at any time. Going Back to a labelled question shows your saved
answers; saving again replaces them.

Order: if data/labelled/queue.csv exists (made by build_queue.py), questions
come in queue order and only queued questions are shown: Arabic first, then an
English sample, then your Haqqi cases. Without a queue, every scraped question
is shown in a fixed random shuffle.

Pre-labelling: if data/labelled/prelabels.csv exists (made by prelabel.py), the
boxes of scraped questions start filled with the LLM's suggestion; Haqqi cases
never do. The suggestion is saved next to your label (llm_problem_type, llm_asker).

Blind check: `streamlit run src/label/label_app.py -- --blind` shows only the
random sample from make_blind_sample.py, never loads the LLM's suggestions, and
saves to data/labelled/blind_labels.csv. Compare with blind_agreement.py.
Check with suggestions: `streamlit run src/label/label_app.py -- --check` shows
the same sample pre-filled with the LLM's suggestions and saves to
data/labelled/check_labels.csv. This is NOT a blind check: report it as
"verified with the LLM's suggestions visible".
Test set: `streamlit run src/label/label_app.py -- --test` shows the 91 RQ1 test
questions (data/labelled/rq1_dev_split.csv) in a fixed mixed order, never loads
the LLM's suggestions, hides the lawyers' answers and saves to
data/labelled/test_labels.csv. These human labels replace the silver labels as
the test set (src/models/common.py).
All modes record how many seconds you spent on each question (seconds_spent).

Arabic is shown right-to-left. The labelling guide (docs/labelling_guide.md) is
shown in the sidebar; edit the file and refresh the page to update it.
"""
import html
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.label.build_queue import QUEUE, all_questions  # noqa: E402
from src.label.prelabel import OUT as PRELABELS  # noqa: E402
from src.label.make_blind_sample import SAMPLE as BLIND_SAMPLE  # noqa: E402

BLIND = "--blind" in sys.argv  # streamlit run src/label/label_app.py -- --blind
CHECK = "--check" in sys.argv  # same sample, with the LLM's suggestions shown
TEST = "--test" in sys.argv    # the RQ1 test questions, labelled blind
SAMPLE_MODE = BLIND or CHECK
LABELS = ("data/labelled/blind_labels.csv" if BLIND else
          "data/labelled/check_labels.csv" if CHECK else
          "data/labelled/test_labels.csv" if TEST else "data/labelled/labelled.csv")
TEST_SPLIT = "data/labelled/rq1_dev_split.csv"
GUIDE = "docs/labelling_guide.md"  # shown in the sidebar
SEED = 42

PROBLEM_TYPES = ["unpaid_wages", "passport_withheld", "gratuity",
                 "unfair_termination", "recruitment_fees", "court_enforcement", "other"]
ASKERS = ["worker", "employer", "unclear"]
# column name -> label shown in the app
FACTS = {"amount": "Amount", "date_mentioned": "Date", "duration": "Duration", "employer": "Employer"}

# `text` + `problem_type` are what tfidf_baseline.py needs
# llm_problem_type / llm_asker: what the LLM suggested (empty if no suggestion),
# kept so you can report how often you changed it
LABEL_COLS = ["question_id", "text", "lang", "law_era", "problem_type", "asker",
              *FACTS, "notes", "source", "llm_problem_type", "llm_asker", "labelled_at", "seconds_spent"]


# --- data helpers (no Streamlit here, so they can be tested on their own) ----

def load_questions(queue_path: str = QUEUE) -> pd.DataFrame:
    """Scraped questions + Haqqi cases, in queue order if there is a queue.
    In blind / check mode: only the sample, in sample order."""
    df = all_questions()
    if TEST:
        # mixed language order, fixed, so the order gives nothing away
        test = pd.read_csv(TEST_SPLIT, dtype=str)[["question_id"]].sample(frac=1, random_state=SEED)
        test["group"] = "test_set"
        return test.merge(df, on="question_id", how="inner")
    if SAMPLE_MODE:
        if not os.path.exists(BLIND_SAMPLE):
            raise SystemExit(f"No {BLIND_SAMPLE}: run python src/label/make_blind_sample.py first.")
        sample = pd.read_csv(BLIND_SAMPLE, encoding="utf-8-sig", dtype=str)[["question_id"]]
        sample["group"] = "blind_check" if BLIND else "check_with_suggestions"
        return sample.merge(df, on="question_id", how="inner")
    if os.path.exists(queue_path):
        queue = pd.read_csv(queue_path, encoding="utf-8-sig", dtype=str)[["question_id", "group"]]
        return queue.merge(df, on="question_id", how="inner")  # keeps queue order
    # no queue: fixed random order, so any prefix you label is a random sample
    df["group"] = "all"
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def load_labels(path: str = LABELS) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=LABEL_COLS)
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def save_label(row: dict, path: str = LABELS) -> None:
    """Add one label, replacing any earlier label for the same question."""
    labels = load_labels(path)
    labels = labels[labels["question_id"] != row["question_id"]]
    labels = pd.concat([labels, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # write to a temporary file first, so a crash never leaves a half-written CSV
    tmp = path + ".tmp"
    labels.reindex(columns=LABEL_COLS).to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def load_prelabels(path: str = PRELABELS) -> dict:
    """LLM suggestions by question_id, without the llm_ prefix (from prelabel.py)."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = df[df["error"] == ""]
    return {r["question_id"]: {k.removeprefix("llm_"): v for k, v in r.items() if k.startswith("llm_")}
            for r in df.to_dict("records")}


def next_unlabelled(ids: list, done: set, start: int) -> int:
    """Position of the first unlabelled question at or after `start`
    (wrapping round to the beginning); len(ids) if everything is labelled."""
    for i in list(range(start, len(ids))) + list(range(0, start)):
        if ids[i] not in done:
            return i
    return len(ids)


# --- display helpers ---------------------------------------------------------

def text_dir(lang: str) -> str:
    return {"ar": "rtl", "en": "ltr"}.get(lang, "auto")  # mixed: browser decides


def show_text(text: str, lang: str, size: str = "1.1rem") -> None:
    """Show text with the right direction (Arabic right-to-left)."""
    st.markdown(
        f'<div dir="{text_dir(lang)}" style="text-align:start; font-size:{size}; '
        f'line-height:1.8">{html.escape(str(text))}</div>',
        unsafe_allow_html=True,
    )


# --- the app -----------------------------------------------------------------

def main():
    st.set_page_config(page_title="Haqqi labelling", layout="centered")
    # text boxes follow the direction of what you type (Arabic employer names etc.)
    st.markdown("<style>input, textarea {unicode-bidi: plaintext;}</style>", unsafe_allow_html=True)

    questions = load_questions()
    labels = load_labels()
    done = set(labels["question_id"])
    prelabels = {} if BLIND or TEST else load_prelabels()  # blind / test: suggestions are never loaded
    if TEST:
        st.error("TEST SET: label from scratch, from the question only. Suggestions and lawyers' answers "
                 "are hidden; answers go to test_labels.csv.")
    if BLIND:
        st.error("BLIND CHECK: label from scratch. The LLM's suggestions are hidden; "
                 "answers go to blind_labels.csv.")
    if CHECK:
        st.warning("CHECK WITH SUGGESTIONS (not blind): the boxes show the LLM's labels; "
                   "answers go to check_labels.csv.")

    # sidebar: filters and progress
    st.sidebar.header("Filter")
    langs = st.sidebar.multiselect("Language", ["en", "ar", "mixed"], default=["en", "ar", "mixed"])
    eras = st.sidebar.multiselect("Law era", ["current", "pre2022"], default=["current", "pre2022"])
    view = questions[questions["lang"].isin(langs) & questions["law_era"].isin(eras)]
    ids = view["question_id"].tolist()

    # start (or restart after a filter change) at the first unlabelled question
    if st.session_state.get("filter") != (tuple(langs), tuple(eras)):
        st.session_state.filter = (tuple(langs), tuple(eras))
        st.session_state.pos = next_unlabelled(ids, done, 0)
    pos = st.session_state.pos

    st.sidebar.header("Progress")
    st.sidebar.write(f"**{len(done & set(ids))} / {len(ids)}** labelled in this filter "
                     f"({len(done)} in total)")
    if len(labels):
        st.sidebar.write(labels.groupby("lang").size().rename("labelled"))
        st.sidebar.write(labels["problem_type"].value_counts().rename("count"))

    if os.path.exists(GUIDE):
        with st.sidebar.expander("Labelling guide", expanded=True):
            with open(GUIDE, encoding="utf-8") as f:
                st.markdown(f.read())

    if pos >= len(ids):
        st.success("Every question in this filter is labelled.")
        if ids and st.button("Back to the last question"):
            st.session_state.pos = len(ids) - 1
            st.rerun()
        return

    q = view.iloc[pos]
    qid = q["question_id"]
    saved = labels[labels["question_id"] == qid]
    saved = saved.iloc[0].to_dict() if len(saved) else {}
    # LLM suggestion: scraped (training) questions only, never Haqqi test cases
    suggestion = prelabels.get(qid, {}) if q["source"] != "haqqi" else {}
    start = saved or suggestion  # what the boxes are filled with
    # when this question was first shown, for seconds_spent
    st.session_state.setdefault(f"shown_at_{qid}", time.time())

    link = f" · [original]({q['url']})" if q["url"] and not TEST else ""
    asked = f" · asked {q['date']}" if q["date"] else ""
    st.caption(f"Question {pos + 1} of {len(ids)} · {q['group']} · id {qid} · {q['lang']} · "
               f"{q['law_era']}{asked}{link}")
    if saved:
        st.info("Already labelled. Saving again replaces your earlier answer.")
    elif suggestion:
        st.warning("Pre-filled with the LLM's suggestion. Check and correct every field before saving.")

    if q["title"]:
        show_text(q["title"], q["lang"], size="1.4rem")
        st.write("")
    show_text(q["text_masked"].removeprefix(str(q["title"])).lstrip(". "), q["lang"])

    answers = json.loads(q["answers"])
    if answers and not TEST:  # test set: label from the question only
        with st.expander("Lawyers' answers (for context only)"):
            for a in answers:
                show_text(a, q["lang"], size="0.95rem")
                st.divider()

    st.subheader("Labels")
    # widget keys include the question id, so each question starts fresh
    problem_type = st.segmented_control(
        "Problem type", PROBLEM_TYPES, key=f"problem_type_{qid}",
        default=start.get("problem_type") or None,
        format_func=lambda s: s.replace("_", " "),
    )
    asker = st.radio(
        "Who is asking?", ASKERS, horizontal=True, key=f"asker_{qid}",
        index=ASKERS.index(start["asker"]) if start.get("asker") in ASKERS else None,
    )

    st.markdown("**Facts stated in the message** (leave empty if not stated)")
    cols = st.columns(2)
    facts = {}
    for i, (col, shown) in enumerate(FACTS.items()):
        facts[col] = cols[i % 2].text_input(shown, value=start.get(col, ""), key=f"{col}_{qid}").strip()
    notes = st.text_input("Notes (optional)", value=start.get("notes", ""), key=f"notes_{qid}").strip()

    back, skip, save = st.columns([1, 1, 2])
    if back.button("← Back", disabled=pos == 0, use_container_width=True):
        st.session_state.pos = pos - 1
        st.rerun()
    if skip.button("Skip →", use_container_width=True):
        st.session_state.pos = pos + 1
        st.rerun()
    if save.button("Save & next", type="primary", use_container_width=True):
        if not problem_type or not asker:
            st.error("Choose a problem type and who is asking before saving.")
        else:
            save_label({
                "question_id": qid,
                "text": q["text_masked"],
                "lang": q["lang"],
                "law_era": q["law_era"],
                "problem_type": problem_type,
                "asker": asker,
                **facts,
                "notes": notes,
                "source": q["source"],
                "llm_problem_type": suggestion.get("problem_type", ""),
                "llm_asker": suggestion.get("asker", ""),
                "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seconds_spent": round(time.time() - st.session_state[f"shown_at_{qid}"]),
            })
            st.session_state.pos = next_unlabelled(ids, done | {qid}, pos + 1)
            st.rerun()


if __name__ == "__main__":  # `streamlit run` runs the file as __main__
    main()
