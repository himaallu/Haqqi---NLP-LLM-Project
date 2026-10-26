"""
Step 3 of scraping: turn the saved HTML pages into one clean CSV (BeautifulSoup).

Usage:
    python src/scrape/parse_pages.py

Input:  data/raw/html/*.html  and  data/raw/links.csv
Output: data/raw/questions.csv

How it finds the content:
  1. The page itself (CSS selectors in SELECTORS, checked against saved pages
     on 22 Oct 2026): title, full question text and the lawyers' answers.
  2. schema.org JSON-LD (<script type="application/ld+json">, type QAPage) for
     the date. We do NOT take the text from JSON-LD: on this site it is a short
     search-engine summary ("Q: <start of question>... A: <start of answer>..."),
     so it is truncated and mixes in the lawyer's answer.
  Pages without JSON-LD: the date is read from the question block instead.
  If the page selectors find no question text, the JSON-LD summary is used and
  the row is marked parse_method = "jsonld_summary" so you can drop or check it.

Run it on 5 pages first and check the CSV by eye:
    python src/scrape/parse_pages.py --limit 5
"""
import argparse
import csv
import glob
import json
import os
import re

from bs4 import BeautifulSoup

HTML_DIR = "data/raw/html"
LINKS = "data/raw/links.csv"
OUT = "data/raw/questions.csv"

# CSS selectors for legaladviceme.com question pages (checked by hand)
SELECTORS = {
    "title": "h1.head-block__title",
    "question_body": "div.head-block__text",
    "answer_text": "div.answer-block__text",
}
JSONLD_TITLE_SUFFIX = " - Q&A - Legal Advice Middle East"
# a few pages have no JSON-LD; their date is printed in the question block, e.g. "Mar 5, 2024"
PAGE_DATE = re.compile(r"\b[A-Z][a-z]{2} \d{1,2}, \d{4}\b")

FIELDS = ["question_id", "url", "title", "question", "date", "n_answers",
          "answers", "answer_laws", "sidebar_laws", "parse_method"]

# Links to laws look like /legislation/<law id>/<slug>. The id is stable; the slug
# and http/https vary. 159 = Federal Decree-Law No. 33 of 2021 (labour law).
LAW_LINK = re.compile(r"/legislation/(\d+)/([^/?#]+)")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def html_to_text(html: str) -> str:
    return clean(BeautifulSoup(html or "", "html.parser").get_text(" "))


def jsonld_question(soup):
    """The schema.org Question object from the page's JSON-LD, or None."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data])
        for item in items:
            q = item.get("mainEntity", item) if isinstance(item, dict) else None
            if isinstance(q, dict) and q.get("@type") == "Question":
                return q
    return None


def from_page(soup):
    """Title, full question and answers from the visible page."""
    title = soup.select_one(SELECTORS["title"])
    body = soup.select_one(SELECTORS["question_body"])
    answers = [clean(a.get_text(" ")) for a in soup.select(SELECTORS["answer_text"])]
    return {
        "title": clean(title.get_text(" ")) if title else "",
        "question": clean(body.get_text(" ")) if body else "",
        "answers": [a for a in answers if a],
    }


def answer_laws(soup):
    """Law links written into the lawyers' answers.

    One entry per link: law_id (from the URL), law (URL slug, a readable law
    name) and text (what the lawyer wrote, sometimes an article, e.g.
    "Article 44, Clause 8").
    """
    out = []
    for a in soup.select("div.answer-block a[href*='/legislation/']"):
        m = LAW_LINK.search(a["href"])
        if m:
            out.append({"law_id": m.group(1), "law": m.group(2), "text": clean(a.get_text(" "))})
    return out


def sidebar_laws(soup):
    """Titles in the site's own "related legislation" box (same on most pages)."""
    out = []
    for a in soup.select("div.related-list a[href*='/legislation/']"):
        t = clean(a.get_text(" "))
        if t and t not in out:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    urls = {}
    if os.path.exists(LINKS):
        with open(LINKS, newline="", encoding="utf-8") as f:
            urls = {r["question_id"]: r["url"] for r in csv.DictReader(f)}

    files = sorted(glob.glob(f"{HTML_DIR}/*.html"))
    if args.limit:
        files = files[: args.limit]

    rows, methods = [], {}
    for path in files:
        qid = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        rec = from_page(soup)
        q = jsonld_question(soup) or {}
        rec["date"] = q.get("dateCreated", "")
        if not rec["date"]:
            block = soup.select_one("div.head-block")
            m = PAGE_DATE.search(block.get_text(" ")) if block else None
            rec["date"] = m.group(0) if m else ""
        rec["parse_method"] = "page"
        if not rec["question"] and q:  # fall back to the (truncated) summary
            rec["title"] = rec["title"] or clean(q.get("name", "")).removesuffix(JSONLD_TITLE_SUFFIX)
            rec["question"] = html_to_text(q.get("text", ""))
            rec["parse_method"] = "jsonld_summary"
        methods[rec["parse_method"]] = methods.get(rec["parse_method"], 0) + 1
        rows.append({
            "question_id": qid,
            "url": urls.get(qid, ""),
            "title": rec["title"],
            "question": rec["question"],
            "date": rec["date"],
            "n_answers": len(rec["answers"]),
            "answers": json.dumps(rec["answers"], ensure_ascii=False),
            "answer_laws": json.dumps(answer_laws(soup), ensure_ascii=False),
            "sidebar_laws": json.dumps(sidebar_laws(soup), ensure_ascii=False),
            "parse_method": rec["parse_method"],
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel shows Arabic
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    empty = sum(1 for r in rows if not r["question"])
    print(f"Parsed {len(rows)} pages -> {OUT}  (methods: {methods}, empty question text: {empty})")
    if empty:
        print("Some pages had no question text: check SELECTORS at the top of this file.")


if __name__ == "__main__":
    main()
